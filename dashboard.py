import streamlit as st

st.set_page_config(
    page_title="CYBER THREAT FINDER",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
.block-container {
    max-width: 1150px;
    padding-top: 3rem;
    padding-bottom: 2rem;
}

.hero {
    text-align: center;
    padding: 20px 10px 35px;
}

.hero h1 {
    font-size: 44px;
    font-weight: 850;
    letter-spacing: 1px;
}

.hero p {
    font-size: 17px;
    opacity: 0.75;
}

.nav-card {
    border: 1px solid rgba(128, 128, 128, 0.35);
    border-radius: 18px;
    padding: 28px;
    min-height: 210px;
    margin-bottom: 18px;
}

.nav-card h2 {
    font-size: 25px;
    margin-bottom: 15px;
}

.nav-card p {
    line-height: 1.7;
    opacity: 0.85;
}

div.stButton > button {
    width: 100%;
    min-height: 58px;
    border-radius: 12px;
    font-size: 17px;
    font-weight: 750;
    border: 1px solid #64748b;
}

div.stButton > button:hover {
    border-color: #38bdf8;
    color: #38bdf8;
}
</style>

<div class="hero">
    <h1>🛡️ CYBER THREAT FINDER</h1>
    <p>AI-POWERED CYBERSECURITY & IoT THREAT ANALYSIS</p>
</div>
""", unsafe_allow_html=True)

st.divider()

left, right = st.columns(2, gap="large")

with left:
    st.markdown("""
    <div class="nav-card">
        <h2>🛡️ CYBER THREAT DASHBOARD</h2>
        <p>
            Explore the existing machine-learning cybersecurity
            framework, threat categories, risk scores, model metrics,
            and sample security alerts.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("OPEN CYBER THREAT DASHBOARD  →"):
        st.switch_page("pages/1_Threat_Dashboard.py")

with right:
    st.markdown("""
    <div class="nav-card">
        <h2>🌐 IOT THREAT DETECTION</h2>
        <p>
            Analyze IoT traffic using the CICIoT2023-trained model.
            Upload a compatible CSV file to generate threat predictions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("OPEN IOT THREAT DETECTION  →"):
        st.switch_page("pages/2_IoT_Threat_Detection.py")

st.divider()

st.caption(
    "ACADEMIC RESEARCH PROTOTYPE  |  PYTHON  |  MACHINE LEARNING  |  STREAMLIT"
)
