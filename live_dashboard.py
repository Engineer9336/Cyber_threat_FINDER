"""
live_dashboard.py  --  Live IoT Network Threat Finder
Run:  streamlit run live_dashboard.py

Modes
  1. Upload CSV        : browser upload (limit set in .streamlit/config.toml)
  2. Local file path   : for HUGE files (GBs) -- streamed from disk, no upload needed
  3. Live feed (tail)  : follows a CSV that your IoT collector keeps appending to

The CSV is processed in chunks; the dashboard updates after every chunk so you
see threats appear in real time. Memory stays flat regardless of file size.
"""
import io
import os
import tempfile
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from threat_engine import ThreatEngine

st.set_page_config(page_title="Live IoT Threat Finder", page_icon="🛡️", layout="wide")

LEVEL_COLORS = {'LOW': '#2ecc71', 'MEDIUM': '#f39c12', 'HIGH': '#e67e22', 'CRITICAL': '#c0392b'}
LEVELS = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
HIST_BINS = np.linspace(-1.0, -0.2, 41)
IP_COLS = ['src_ip', 'srcip', 'source_ip', 'saddr', 'ip', 'device', 'device_id']


@st.cache_resource(show_spinner="Loading models...")
def get_engine():
    return ThreatEngine()


class Aggregator:
    """Keeps only running aggregates -- never the whole file."""

    def __init__(self):
        self.total = 0
        self.level = {l: 0 for l in LEVELS}
        self.category = {}
        self.anomalies = 0
        self.anomaly_and_threat = 0
        self.timeline = []                     # one row per chunk
        self.hist = np.zeros(len(HIST_BINS) - 1, dtype=np.int64)
        self.top_alerts = pd.DataFrame()
        self.ip_counts = {}
        self.ip_col = None
        self.conf_sum = 0.0
        self.t0 = time.time()

    def update(self, raw: pd.DataFrame, sc: pd.DataFrame):
        n = len(sc)
        row_ids = np.arange(self.total, self.total + n)
        self.total += n
        for l, c in sc['risk_level'].value_counts().items():
            self.level[l] += int(c)
        for k, c in sc['threat_category'].value_counts().items():
            self.category[k] = self.category.get(k, 0) + int(c)
        self.anomalies += int(sc['is_anomaly'].sum())
        threat_mask = sc['risk_level'] != 'LOW'
        self.anomaly_and_threat += int((sc['is_anomaly'] & threat_mask).sum())
        self.hist += np.histogram(sc['anomaly_score'], bins=HIST_BINS)[0]
        self.conf_sum += float(sc['confidence'].sum())
        self.timeline.append({
            'batch': len(self.timeline) + 1, 'rows': n,
            'threats': int(threat_mask.sum()),
            'critical': int((sc['risk_level'] == 'CRITICAL').sum()),
            'high': int((sc['risk_level'] == 'HIGH').sum()),
            'avg_risk': float(sc['risk_score'].mean()),
            'elapsed': time.time() - self.t0,
        })

        # top alerts (keep 300 highest-risk, with row number + original ip/device if any)
        view = sc.loc[threat_mask].copy()
        view.insert(0, 'row', row_ids[threat_mask.values])
        if self.ip_col is None:
            self.ip_col = next((c for c in IP_COLS if c in raw.columns), None)
        if self.ip_col:
            view.insert(1, self.ip_col, raw.loc[view.index, self.ip_col].astype(str).values)
            for ip, c in view[self.ip_col].value_counts().items():
                self.ip_counts[ip] = self.ip_counts.get(ip, 0) + int(c)
            if len(self.ip_counts) > 5000:      # bound memory
                self.ip_counts = dict(sorted(self.ip_counts.items(), key=lambda x: -x[1])[:2000])
        self.top_alerts = (pd.concat([self.top_alerts, view])
                           .sort_values(['risk_score', 'confidence'], ascending=False).head(300))

    @property
    def threats(self):
        return self.total - self.level['LOW']


def build_insights(a: Aggregator) -> list:
    if a.total == 0:
        return []
    out = []
    rate = 100 * a.threats / a.total
    sev = 'high' if rate > 40 else 'moderate' if rate > 10 else 'low'
    out.append(f"**{a.threats:,} of {a.total:,} flows ({rate:.1f}%)** were flagged as threats -- {sev} threat load.")
    threats_only = {k: v for k, v in a.category.items() if k != 'Normal'}
    if threats_only:
        top = max(threats_only, key=threats_only.get)
        out.append(f"Dominant attack type: **{top}** ({threats_only[top]:,} flows, "
                   f"{100 * threats_only[top] / sum(threats_only.values()):.1f}% of attack-labelled flows).")
    if a.level['CRITICAL']:
        out.append(f"🚨 **{a.level['CRITICAL']:,} CRITICAL** flows (DoS/Worm-class) need immediate review.")
    if a.anomalies:
        out.append(f"Isolation Forest marked {a.anomalies:,} flows as statistical outliers; "
                   f"{a.anomaly_and_threat:,} of them were also classified as threats "
                   f"(rest may be unusual-but-benign or novel behaviour worth a look).")
    tl = pd.DataFrame(a.timeline)
    if len(tl) > 2:
        tl['rate'] = tl['threats'] / tl['rows']
        pk = tl.loc[tl['rate'].idxmax()]
        out.append(f"Peak threat density in batch **#{int(pk['batch'])}** ({100 * pk['rate']:.1f}% of that batch).")
    if a.ip_counts:
        ip, c = max(a.ip_counts.items(), key=lambda x: x[1])
        out.append(f"Most flagged source ({a.ip_col}): **{ip}** with {c:,} threat flows.")
    return out


def render(a: Aggregator, slot, tick: int, status: str):
    with slot.container():
        st.caption(status)
        elapsed = max(time.time() - a.t0, 1e-6)
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Flows analysed", f"{a.total:,}")
        k2.metric("Threats", f"{a.threats:,}", f"{100 * a.threats / max(a.total, 1):.1f}%", delta_color="inverse")
        k3.metric("Critical", f"{a.level['CRITICAL']:,}")
        k4.metric("Anomalies", f"{a.anomalies:,}")
        k5.metric("Speed", f"{a.total / elapsed:,.0f} flows/s")

        tl = pd.DataFrame(a.timeline)
        c1, c2 = st.columns([3, 2])
        with c1:
            st.subheader("📈 Threat timeline (per batch)")
            if len(tl):
                fig = go.Figure()
                fig.add_bar(x=tl['batch'], y=tl['threats'], name='Threats', marker_color='#e74c3c')
                fig.add_bar(x=tl['batch'], y=tl['critical'], name='Critical', marker_color='#7b241c')
                fig.add_scatter(x=tl['batch'], y=tl['avg_risk'], name='Avg risk', yaxis='y2',
                                line=dict(color='#2980b9'))
                fig.update_layout(barmode='overlay', height=330, margin=dict(l=0, r=0, t=10, b=0),
                                  yaxis2=dict(overlaying='y', side='right', range=[0, 100], title='Avg risk'),
                                  legend=dict(orientation='h'))
                st.plotly_chart(fig, width='stretch', key=f"tl{tick}")
        with c2:
            st.subheader("⚠️ Risk levels")
            lv = pd.DataFrame({'level': LEVELS, 'count': [a.level[l] for l in LEVELS]})
            fig = px.pie(lv, names='level', values='count', hole=.55, color='level',
                         color_discrete_map=LEVEL_COLORS)
            fig.update_layout(height=330, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, width='stretch', key=f"pie{tick}")

        c3, c4 = st.columns(2)
        with c3:
            st.subheader("📊 Attack categories")
            cats = pd.Series({k: v for k, v in a.category.items() if k != 'Normal'}).sort_values()
            if len(cats):
                fig = px.bar(x=cats.values, y=cats.index, orientation='h', color_discrete_sequence=['#e74c3c'])
                fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), xaxis_title='Flows', yaxis_title='')
                st.plotly_chart(fig, width='stretch', key=f"cat{tick}")
            else:
                st.success("No threats detected so far.")
        with c4:
            st.subheader("🧪 Anomaly score distribution")
            mid = (HIST_BINS[:-1] + HIST_BINS[1:]) / 2
            fig = px.bar(x=mid, y=a.hist, color_discrete_sequence=['#3498db'])
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                              xaxis_title='Isolation Forest score (lower = more anomalous)', yaxis_title='Flows')
            st.plotly_chart(fig, width='stretch', key=f"an{tick}")

        if a.ip_counts:
            st.subheader(f"🌐 Top flagged sources ({a.ip_col})")
            ips = pd.Series(a.ip_counts).sort_values(ascending=False).head(10)[::-1]
            fig = px.bar(x=ips.values, y=ips.index.astype(str), orientation='h',
                         color_discrete_sequence=['#8e44ad'])
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), xaxis_title='Threat flows', yaxis_title='')
            st.plotly_chart(fig, width='stretch', key=f"ip{tick}")

        st.subheader("🧠 Auto analysis")
        for line in build_insights(a):
            st.markdown(f"- {line}")

        st.subheader("🚨 Highest-risk alerts (top 25)")
        if len(a.top_alerts):
            st.dataframe(a.top_alerts.head(25), width='stretch', hide_index=True)


# ---------------------------------------------------------------------------
def tail_batches(path: str, chunk_rows: int, poll: float):
    """Follow a growing CSV. Yields DataFrames of newly appended rows."""
    with open(path, 'r', newline='') as f:
        header = f.readline()
        while True:
            lines = []
            while len(lines) < chunk_rows:
                line = f.readline()
                if not line or not line.endswith('\n'):   # EOF or half-written line
                    if line:
                        f.seek(f.tell() - len(line.encode()))
                    break
                lines.append(line)
            if lines:
                df = pd.read_csv(io.StringIO(header + ''.join(lines)), skipinitialspace=True)
                df.columns = [c.strip().lower() for c in df.columns]
                yield df
            else:
                time.sleep(poll)
                yield None


# ---------------------------------------------------------------------------
engine = get_engine()

st.title("🛡️ Live IoT Network Threat Finder")
st.caption("Upload network-flow CSVs of any size -- threats, risk levels and analysis update live as data is processed.")

with st.sidebar:
    st.header("⚙️ Input")
    mode = st.radio("Source", ["Upload CSV", "Local file path (huge files)", "Live feed (tail a growing CSV)"])
    chunk = st.select_slider("Batch size (rows)", [1000, 5000, 10000, 20000, 50000, 100000], value=20000)
    delay = st.slider("Replay delay per batch (sec)", 0.0, 3.0, 0.3, 0.1,
                      help="Slow it down to watch the dashboard fill up like a live stream. 0 = max speed.")
    uploaded, path = None, ""
    if mode == "Upload CSV":
        uploaded = st.file_uploader("CSV file", type=["csv"])
    elif mode.startswith("Local"):
        path = st.text_input("Full path to CSV on this machine", placeholder="/data/iot_flows_5GB.csv")
    else:
        path = st.text_input("CSV being appended by your collector", placeholder="/data/live_flows.csv")
        poll = st.slider("Poll interval (sec)", 0.5, 10.0, 2.0, 0.5)
    run = st.button("▶ Start analysis", type="primary", width='stretch')
    st.markdown("---")
    st.markdown(f"**Model:** {engine.meta['best_model_name']}  \n"
                f"**Multiclass model:** {'loaded' if engine.multiclass is not None else 'not found (threats shown as *Unclassified Threat*)'}")
    with st.expander("Required CSV columns"):
        st.code(", ".join(engine.required_cols), language=None)

slot = st.empty()

if not run:
    with slot.container():
        st.info("Left sidebar se CSV choose karke **Start analysis** dabao. "
                "CSV mein UNSW-NB15 jaise flow columns hone chahiye (list sidebar mein hai).")
    st.stop()

# ---- resolve source ---------------------------------------------------------
source = None
if mode == "Upload CSV":
    if uploaded is None:
        st.warning("Pehle CSV upload karo."); st.stop()
    source = uploaded
    head_cols = pd.read_csv(io.BytesIO(uploaded.getvalue()[:200000].rsplit(b'\n', 1)[0]), nrows=1).columns
    uploaded.seek(0)
else:
    if not path or not os.path.isfile(path):
        st.error("File path sahi nahi hai ya file nahi mili."); st.stop()
    source = path
    head_cols = pd.read_csv(path, nrows=1).columns

missing, extra = engine.check_columns(head_cols)
if missing:
    st.warning(f"{len(missing)} model columns CSV mein nahi mile -- unko 0 maana jayega "
               f"(accuracy kam ho sakti hai): {', '.join(missing[:12])}{' ...' if len(missing) > 12 else ''}")
if len(missing) > len(engine.required_cols) * 0.5:
    st.error("Zyada tar required columns missing hain -- ye CSV is model ke format ka nahi hai."); st.stop()

agg = Aggregator()
out_path = os.path.join(tempfile.gettempdir(), f"threat_results_{int(time.time())}.csv")
wrote_header = False
tick = 0

def save(raw, sc):
    global wrote_header
    keep = [c for c in raw.columns if c in IP_COLS or c in ('id', 'label', 'attack_cat', 'timestamp', 'stime')]
    res = pd.concat([raw[keep], sc], axis=1)
    res.to_csv(out_path, mode='a', header=not wrote_header, index=False)
    wrote_header = True

try:
    if mode.startswith("Live"):
        stop_note = "🔴 LIVE -- feed follow ho raha hai. Rokne ke liye browser mein Stop dabao ya page refresh karo."
        for raw in tail_batches(path, chunk, poll):
            if raw is None:
                render(agg, slot, tick, f"{stop_note}  (naya data wait kar raha hai...)"); tick += 1
                continue
            sc = engine.score(raw); agg.update(raw, sc); save(raw, sc)
            render(agg, slot, tick, stop_note); tick += 1
    else:
        for raw, sc in engine.stream_csv(source, chunk):
            agg.update(raw, sc); save(raw, sc)
            render(agg, slot, tick, f"⏳ Processing... batch {len(agg.timeline)}"); tick += 1
            if delay:
                time.sleep(delay)
        render(agg, slot, tick, f"✅ Done -- {agg.total:,} flows in {time.time() - agg.t0:.1f}s")
except Exception as e:
    st.error(f"Processing error: {e}")

if agg.total and os.path.isfile(out_path):
    with open(out_path, 'rb') as f:
        st.download_button("⬇️ Download full scored results (CSV)", f, file_name="threat_results.csv", mime="text/csv")
