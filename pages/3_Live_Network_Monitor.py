"""
Live Network Monitor page for the existing Cyber Threat Finder Streamlit app.

Put this file at:
    pages/3_Live_Network_Monitor.py

IMPORTANT:
- Live packet capture runs only where this Streamlit process runs.
- Streamlit Cloud generally cannot capture traffic from your personal laptop/Wi-Fi.
- For actual local monitoring, run the app on the computer/network you are authorized to monitor.
- Alerts are heuristic indicators, not proof of malicious activity.
"""

from collections import defaultdict, deque
from datetime import datetime
import queue
import time

import pandas as pd
import streamlit as st

try:
    from scapy.all import AsyncSniffer, IP, IPv6, TCP, UDP, DNS
    SCAPY_OK = True
    SCAPY_ERROR = ""
except Exception as exc:
    SCAPY_OK = False
    SCAPY_ERROR = str(exc)

st.markdown("""
<style>
.stApp {
  background: radial-gradient(circle at 10% 0%, rgba(0,229,255,.07), transparent 28%), #07111f;
  color: #e6f1ff;
}
[data-testid="stSidebar"] {background:#081625;border-right:1px solid #17394e;}
.block-container {max-width:1450px;padding-top:1.4rem;}
h1,h2,h3 {color:#e9f7ff!important;}
div[data-testid="stMetric"] {
  background:linear-gradient(145deg,#10263a,#091827);
  border:1px solid #17435b;border-top:2px solid #00d9f5;
  padding:14px 16px;border-radius:12px;
}
.live-tag {color:#00e5ff;font-weight:800;letter-spacing:.14em;font-size:.78rem;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="live-tag">CYBER SECURITY OPERATIONS • NETWORK SENSOR</div>', unsafe_allow_html=True)
st.title("📡 Live Network Threat Monitor")
st.caption("Packet-metadata monitoring and heuristic alerts. Monitor only networks you own or are authorized to inspect.")

if not SCAPY_OK:
    st.error("Scapy is missing or could not load. Add `scapy` to requirements.txt and redeploy.")
    st.code("scapy")
    st.caption(f"Import detail: {SCAPY_ERROR}")
    st.stop()

# Queue is shared with Scapy's capture thread; UI/session-state is only updated on the main thread.
if "live_packet_queue" not in st.session_state:
    st.session_state.live_packet_queue = queue.Queue()
if "live_packet_rows" not in st.session_state:
    st.session_state.live_packet_rows = deque(maxlen=2000)
if "live_alert_rows" not in st.session_state:
    st.session_state.live_alert_rows = deque(maxlen=300)
if "live_sniffer" not in st.session_state:
    st.session_state.live_sniffer = None
if "live_seen_syn" not in st.session_state:
    st.session_state.live_seen_syn = defaultdict(deque)
if "live_seen_ports" not in st.session_state:
    st.session_state.live_seen_ports = defaultdict(deque)
if "live_total_packets" not in st.session_state:
    st.session_state.live_total_packets = 0
if "live_started_at" not in st.session_state:
    st.session_state.live_started_at = None

packet_queue = st.session_state.live_packet_queue

def packet_callback(packet):
    try:
        src = dst = "Unknown"
        sport = dport = None
        protocol = "Other"
        flags = ""
        if IP in packet:
            src, dst = packet[IP].src, packet[IP].dst
        elif IPv6 in packet:
            src, dst = packet[IPv6].src, packet[IPv6].dst

        if TCP in packet:
            protocol = "TCP"
            sport, dport = int(packet[TCP].sport), int(packet[TCP].dport)
            flags = str(packet[TCP].flags)
        elif UDP in packet:
            protocol = "UDP"
            sport, dport = int(packet[UDP].sport), int(packet[UDP].dport)
        elif packet.haslayer("ICMP"):
            protocol = "ICMP"

        dns_name = ""
        if DNS in packet and packet[DNS].qd is not None:
            try:
                qname = packet[DNS].qd.qname
                dns_name = qname.decode("utf-8", errors="ignore").rstrip(".") if isinstance(qname, bytes) else str(qname).rstrip(".")
            except Exception:
                pass

        packet_queue.put({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "epoch": time.time(),
            "source": src, "destination": dst, "protocol": protocol,
            "src_port": sport, "dst_port": dport, "tcp_flags": flags,
            "bytes": int(len(packet)), "dns_query": dns_name,
        })
    except Exception:
        pass

def add_alert(row, severity, title, detail):
    st.session_state.live_alert_rows.appendleft({
        "time": row["time"], "severity": severity, "source": row["source"],
        "alert": title, "details": detail,
    })

def inspect_row(row):
    now, src = row["epoch"], row["source"]
    if row["protocol"] == "TCP" and "S" in row["tcp_flags"] and "A" not in row["tcp_flags"]:
        syn = st.session_state.live_seen_syn[src]
        syn.append(now)
        while syn and now - syn[0] > 10:
            syn.popleft()
        if len(syn) == 35:
            add_alert(row, "High", "Possible SYN burst", "35 TCP SYN attempts from this source in 10 seconds. Investigate; this is not proof of an attack.")

        ports = st.session_state.live_seen_ports[src]
        if row["dst_port"] is not None:
            ports.append((now, row["dst_port"]))
        while ports and now - ports[0][0] > 20:
            ports.popleft()
        distinct = len({port for _, port in ports})
        if distinct == 20:
            add_alert(row, "High", "Possible port scan", "20 distinct destination ports in 20 seconds. Investigate if unexpected.")

    if row["dst_port"] in {23, 445, 3389}:
        add_alert(row, "Low", "Sensitive service traffic", f"Traffic to destination port {row['dst_port']}; verify it is expected.")

with st.sidebar:
    st.header("Sensor controls")
    st.warning("Only monitor systems and networks you are authorized to inspect.")
    refresh = st.slider("Refresh interval", 1, 5, 2, help="How often the dashboard refreshes while monitoring.")
    st.caption("Uses the default capture interface of the machine running this app.")
    start = st.button("▶ Start monitoring", type="primary", use_container_width=True)
    stop = st.button("■ Stop monitoring", use_container_width=True)
    clear = st.button("Clear session", use_container_width=True)

if clear:
    st.session_state.live_packet_rows.clear()
    st.session_state.live_alert_rows.clear()
    st.session_state.live_seen_syn.clear()
    st.session_state.live_seen_ports.clear()
    st.session_state.live_total_packets = 0

if start:
    sniffer = st.session_state.live_sniffer
    if sniffer is None or not getattr(sniffer, "running", False):
        try:
            sniffer = AsyncSniffer(prn=packet_callback, store=False)
            sniffer.start()
            st.session_state.live_sniffer = sniffer
            st.session_state.live_started_at = datetime.now()
        except Exception as exc:
            st.session_state.live_sniffer = None
            st.error(f"Packet capture could not start: {exc}")
            st.info("Local Windows capture may require Npcap and Administrator privileges. Streamlit Cloud cannot capture your laptop's Wi-Fi traffic.")
    else:
        st.info("Monitoring is already running.")

if stop and st.session_state.live_sniffer is not None:
    try:
        if st.session_state.live_sniffer.running:
            st.session_state.live_sniffer.stop()
    except Exception as exc:
        st.warning(f"Capture stop warning: {exc}")
    st.session_state.live_sniffer = None

drained = 0
while drained < 1200:
    try:
        row = packet_queue.get_nowait()
    except queue.Empty:
        break
    st.session_state.live_packet_rows.appendleft(row)
    st.session_state.live_total_packets += 1
    inspect_row(row)
    drained += 1

sniffer = st.session_state.live_sniffer
running = bool(sniffer is not None and getattr(sniffer, "running", False))
st.markdown(f"**Sensor status:** {'🟢 MONITORING' if running else '⚪ STOPPED'}")
if st.session_state.live_started_at:
    st.caption("Session started: " + st.session_state.live_started_at.strftime("%Y-%m-%d %H:%M:%S"))

packets = pd.DataFrame(list(st.session_state.live_packet_rows))
alerts = pd.DataFrame(list(st.session_state.live_alert_rows))
m1, m2, m3, m4 = st.columns(4)
m1.metric("Packets captured", f"{st.session_state.live_total_packets:,}")
m2.metric("Unique sources", f"{packets['source'].nunique() if not packets.empty else 0:,}")
m3.metric("Session alerts", f"{len(alerts):,}")
m4.metric("Bytes in buffer", f"{int(packets['bytes'].sum()) if not packets.empty else 0:,}")

left, right = st.columns(2, gap="large")
with left:
    st.subheader("📊 Protocol distribution")
    if not packets.empty:
        st.bar_chart(packets["protocol"].value_counts(), use_container_width=True)
    else:
        st.info("Protocol chart populates after packet capture starts.")
with right:
    st.subheader("🌐 Top destination ports")
    if not packets.empty:
        ports = packets.dropna(subset=["dst_port"])
        if not ports.empty:
            st.bar_chart(ports["dst_port"].astype(int).value_counts().head(10), use_container_width=True)
        else:
            st.info("No TCP/UDP destination ports captured yet.")
    else:
        st.info("Port chart populates after packet capture starts.")

st.subheader("🚨 Alert queue")
if not alerts.empty:
    st.dataframe(alerts, use_container_width=True, hide_index=True)
else:
    st.success("No heuristic alerts in this session. This does not prove the network is threat-free.")

st.subheader("🧾 Recent packet metadata")
if not packets.empty:
    cols = ["time", "source", "destination", "protocol", "src_port", "dst_port", "tcp_flags", "bytes", "dns_query"]
    st.dataframe(packets[cols].head(100), use_container_width=True, hide_index=True)
    st.download_button("Download packet metadata CSV", packets[cols].to_csv(index=False).encode("utf-8"),
                       "live_packet_metadata.csv", "text/csv")
else:
    st.caption("No packets captured yet.")

st.caption("Prototype limitation: heuristic alerts are triage signals, not confirmed attacks. Payloads are not inspected, and this page does not yet invoke the saved XGBoost model. The capture source is the machine hosting this Streamlit process.")

if running:
    time.sleep(refresh)
    st.rerun()
