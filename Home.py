import streamlit as st

st.set_page_config(
    page_title="Cyber Threat Finder",
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    max-width: 1200px;
}
.hero {
    text-align: center;
    padding: 25px 10px 35px 10px;
}
.hero h1 {
    font-size: 42px;
    font-weight: 800;
}
.hero p {
    font-size: 18px;
    color: #9CA3AF;
}
div.stButton > button {
    width: 100%;
    min-height: 65px;
    border-radius: 12px;
    font-size: 19px;
    font-weight: 700;
    border: 1px solid #64748b;
    transition: 0.2s;
}
div.stButton > button:hover {
    border-color: #38bdf8;
    color: #38bdf8;
}
.card {
    padding: 28px;
    border: 1px solid rgba(128,128,128,0.35);
    border-radius: 18px;
    min-height: 230px;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🛡️ CYBER THREAT FINDER</h1>
    <p>AI-POWERED CYBERSECURITY & IoT THREAT ANALYSIS</p>
</div>
""", unsafe_allow_html=True)

st.divider()

left, right = st.columns(2, gap="large")

with left:
    st.markdown("""
    <div class="card">
        <h2>🛡️ DASHBOARD</h2>
        <p>Explore the existing AI-based cyber threat detection system,
        model insights, attack analysis and risk assessment.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("OPEN DASHBOARD  →", key="open_dashboard"):
        st.switch_page("dashboard.py")

with right:
    st.markdown("""
    <div class="card">
        <h2>🌐 IOT THREAT DETECTION</h2>
        <p>Analyze IoT network traffic using the CICIoT2023-trained
        machine learning model and upload CSV files for predictions.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("OPEN IOT DETECTION  →", key="open_iot"):
        st.switch_page("pages/2_IoT_Threat_Detection.py")

st.divider()

st.caption(
    "Academic Research Prototype | Python | Machine Learning | "
    "XGBoost | Streamlit"
)
