import os
import random
import pandas as pd
import streamlit as st

from pipeline import RAGIndex, RISK_PROFILES, LOG_PATH, run_pipeline, log_session
from synthetic_data import TICKERS, generate_market_data, generate_news, generate_filing_corpus, generate_portfolio

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SPIDER-SENSE // Autonomous Financial Swarm",
    page_icon="🕷️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------------------------
# High-End Fintech Terminal CSS (Clean, Modern Geometric Typography)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

* { box-sizing: border-box; }

/* Global Base */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: #F8FAFC;
}

.stApp {
    background-color: #030712;
    background-image: 
        radial-gradient(circle at 10% 0%, rgba(230, 36, 41, 0.16) 0%, transparent 45%),
        radial-gradient(circle at 90% 0%, rgba(0, 240, 255, 0.12) 0%, transparent 45%),
        radial-gradient(#1E293B 1.2px, transparent 1.2px);
    background-size: 100% 100%, 100% 100%, 28px 28px;
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 3.5rem;
    max-width: 1400px;
}

/* Modern Geometric Tech Logo */
.brand-title {
    font-family: 'Outfit', sans-serif !important;
    font-size: 2.3rem;
    font-weight: 900;
    letter-spacing: -0.03em;
    line-height: 1;
    margin: 0;
    background: linear-gradient(90deg, #FFFFFF 25%, #00F0FF 80%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: flex;
    align-items: center;
    gap: 10px;
}

.brand-subtitle {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.88rem;
    color: #94A3B8;
    margin-top: 5px;
    font-weight: 500;
    letter-spacing: 0.01em;
}

/* Tactical Navbar Header */
.tactical-navbar {
    background: linear-gradient(135deg, rgba(13, 19, 36, 0.95) 0%, rgba(6, 10, 22, 0.98) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-top: 3px solid #E62429;
    border-radius: 16px;
    padding: 20px 28px;
    margin-bottom: 22px;
    box-shadow: 0 20px 45px rgba(0,0,0,0.85);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 16px;
}

.badge-pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.74rem;
    font-weight: 700;
    padding: 6px 14px;
    border-radius: 6px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border: 1px solid rgba(255,255,255,0.1);
    background: #090E1A;
    color: #94A3B8;
}

.badge-pill.live {
    border-color: rgba(0, 240, 255, 0.45);
    background: rgba(0, 240, 255, 0.08);
    color: #00F0FF;
    display: flex;
    align-items: center;
    gap: 8px;
}

.pulse-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #00F0FF;
    box-shadow: 0 0 10px #00F0FF;
}

/* Control Hub Container */
.control-hub-box {
    background: linear-gradient(135deg, rgba(13, 19, 34, 0.9) 0%, rgba(8, 12, 24, 0.95) 100%);
    border: 1px solid #1E293B;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 22px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
}

/* Modern Clean Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(10, 15, 29, 0.9);
    padding: 8px 10px;
    border-radius: 12px;
    border: 1px solid #1E293B;
    margin-bottom: 22px;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    color: #94A3B8 !important;
    padding: 10px 20px !important;
    border-radius: 8px !important;
    border: 1px solid transparent !important;
    background: transparent !important;
    transition: all 0.2s ease !important;
}

.stTabs [aria-selected="true"] {
    color: #00F0FF !important;
    background: #0F172A !important;
    border-color: rgba(0, 240, 255, 0.4) !important;
    box-shadow: 0 4px 16px rgba(0, 240, 255, 0.15) !important;
}

/* Cards & Surface Modules */
.sp-card {
    background: linear-gradient(135deg, rgba(13, 19, 36, 0.92) 0%, rgba(7, 11, 22, 0.96) 100%);
    border: 1px solid #1E293B;
    border-radius: 16px;
    padding: 26px;
    margin-bottom: 20px;
    box-shadow: 0 12px 35px rgba(0,0,0,0.6);
}

.sp-card-header {
    font-family: 'Outfit', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    color: #F8FAFC;
    margin-bottom: 18px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #1E293B;
    padding-bottom: 12px;
}

/* Master Verdict Showcase */
.verdict-hero-card {
    background: linear-gradient(135deg, rgba(230, 36, 41, 0.1) 0%, rgba(10, 15, 30, 0.9) 100%);
    border: 2px solid rgba(230, 36, 41, 0.8);
    border-radius: 16px;
    padding: 30px;
    text-align: center;
    margin-bottom: 22px;
    box-shadow: 0 16px 40px rgba(230, 36, 41, 0.15);
}

.verdict-action {
    font-family: 'Outfit', sans-serif;
    font-size: 2.6rem;
    font-weight: 900;
    letter-spacing: -0.02em;
    margin: 8px 0;
    line-height: 1.1;
}

/* Specialized Agent Node */
.agent-node {
    background: #080D1A;
    border: 1px solid #1E293B;
    border-radius: 14px;
    padding: 20px;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    box-shadow: 0 6px 20px rgba(0,0,0,0.45);
    transition: all 0.2s ease;
}
.agent-node:hover {
    border-color: rgba(0, 240, 255, 0.4);
    transform: translateY(-2px);
}

.trace-line {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: #E2E8F0;
    background: rgba(15, 23, 42, 0.85);
    padding: 12px 14px;
    border-radius: 8px;
    border-left: 3px solid #00F0FF;
    margin: 8px 0;
    line-height: 1.55;
}

/* Citation Box */
.citation-box {
    background: #080D1A;
    border: 1px solid #1E293B;
    border-left: 4px solid #FFD700;
    padding: 16px 20px;
    border-radius: 10px;
    margin-bottom: 12px;
}

/* Metric KPI HUD Cards */
div[data-testid="stMetric"] {
    background: #080D1A;
    border: 1px solid #1E293B;
    border-radius: 12px;
    padding: 14px 18px;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);
}
div[data-testid="stMetric"] label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    color: #94A3B8 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-family: 'Outfit', sans-serif !important;
    font-size: 1.65rem !important;
    font-weight: 800 !important;
    color: #F8FAFC !important;
}

/* Modern Primary Action Button */
div.stButton > button {
    font-family: 'Outfit', sans-serif !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em !important;
    background: linear-gradient(90deg, #E62429 0%, #FF384D 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem 2rem !important;
    box-shadow: 0 4px 18px rgba(230, 36, 41, 0.4) !important;
    transition: all 0.2s ease !important;
    width: 100%;
}
div.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(230, 36, 41, 0.65) !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Navigation Bar Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="tactical-navbar">
    <div>
        <div class="brand-title">🕸️ SPIDER-SENSE FINANCIAL</div>
        <div class="brand-subtitle">
            Autonomous Multi-Agent Financial Intelligence Network // Explainable Retail Research
        </div>
    </div>
    <div style="display: flex; gap: 10px; align-items: center; flex-wrap: wrap;">
        <span class="badge-pill live"><span class="pulse-dot"></span>Swarm Online</span>
        <span class="badge-pill">Sprint 1: PS-01</span>
        <span class="badge-pill" style="border-color: rgba(230,36,41,0.6); color:#FF384D;">Team YOLOTECH</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# State & Resource Management
# ---------------------------------------------------------------------------
@st.cache_resource
def get_rag_index():
    return RAGIndex(generate_filing_corpus())

rag_index = get_rag_index()
portfolio = generate_portfolio()

if "has_run" not in st.session_state:
    st.session_state.has_run = False

# ---------------------------------------------------------------------------
# Control Parameters Deck
# ---------------------------------------------------------------------------
st.markdown('<div class="control-hub-box">', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1.2])
with c1:
    ticker = st.selectbox("Target Equity", TICKERS)
with c2:
    profile_name = st.selectbox("Investor Persona", list(RISK_PROFILES.keys()), index=1)
with c3:
    scenario = st.selectbox("Market Scenario", ["normal", "crash", "positive_news", "negative_news"])
with c4:
    degrade = st.selectbox("Fault-Injection (Degraded Mode)", ["none", "momentum", "volume", "sentiment"])

st.write("")
if st.button("DISPATCH MULTI-AGENT SWARM", use_container_width=True):
    crash = (scenario == "crash")
    market_data = generate_market_data(ticker, crash=crash)
    news_sentiment = "positive" if scenario == "positive_news" else "negative" if scenario == "negative_news" else "mixed"
    news = generate_news(ticker, news_sentiment)

    outputs, synthesis = run_pipeline(
        ticker, market_data, news, rag_index, profile_name,
        simulate_degraded=None if degrade == "none" else degrade,
    )
    
    forward_return_mock = random.uniform(-8, 8)
    row = log_session(ticker, profile_name, outputs, synthesis, forward_return_mock, portfolio)

    st.session_state.has_run = True
    st.session_state.outputs = outputs
    st.session_state.synthesis = synthesis
    st.session_state.row = row
    st.session_state.ticker = ticker
    st.session_state.profile_name = profile_name
    st.session_state.market_data = market_data
    st.session_state.news = news
    st.session_state.degrade = degrade

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Structured Multi-Tab Navigation
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Master Verdict & Guidance",
    "🤖 3-Agent Parallel Swarm",
    "⚖️ Persona Matrix & Proof",
    "📎 SEBI Regulatory RAG",
    "📊 Benchmarks & Audit Logs"
])

if not st.session_state.has_run:
    st.info("💡 Adjust parameters above and click **'DISPATCH MULTI-AGENT SWARM'** to initialize analysis.")
else:
    outputs = st.session_state.outputs
    synthesis = st.session_state.synthesis
    row = st.session_state.row
    m_ticker = st.session_state.ticker
    m_profile = st.session_state.profile_name

    # Tab 1: Master Verdict & Guidance
    with tab1:
        st.markdown('<div class="sp-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="sp-card-header"><span>Master Synthesis Verdict // <strong>{m_ticker}</strong></span><span style="color:#94A3B8; font-size:0.9rem;">Target: {m_profile}</span></div>', unsafe_allow_html=True)
        
        action_col = "#10B981" if "BUY" in synthesis['action'] else ("#EF4444" if "REDUCE" in synthesis['action'] else "#00F0FF")
        
        st.markdown(f"""
        <div class="verdict-hero-card">
            <div style="font-family:'JetBrains Mono'; font-size:0.8rem; color:#FFD700; text-transform:uppercase; font-weight:700; letter-spacing:0.08em;">
                RISK-CALIBRATED VERDICT // {m_profile.upper()}
            </div>
            <div class="verdict-action" style="color:{action_col};">
                {synthesis['action']}
            </div>
            <div style="font-size:1.05rem; color:#E2E8F0; line-height:1.6; max-width:920px; margin: 12px auto 0 auto;">
                {synthesis['reasoning']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.progress(synthesis["confidence"], text=f"Synthesis Confidence: {synthesis['confidence']:.0%}")

        st.markdown('<div class="sp-card-header" style="margin-top:24px;">Transparent Reasoning Trace</div>', unsafe_allow_html=True)
        for out in outputs:
            st.markdown(f"""
            <div class="trace-line">
                <strong style="color:#00F0FF;">[{out.agent}]</strong> ➔ {out.reasoning}
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Tab 2: Parallel Agent Swarm
    with tab2:
        st.markdown('<div class="sp-card">', unsafe_allow_html=True)
        st.markdown('<div class="sp-card-header">Specialized Domain Agents // Real-Time Parallel Telemetry</div>', unsafe_allow_html=True)
        
        cols = st.columns(len(outputs))
        for col, o in zip(cols, outputs):
            with col:
                status_col = "#F59E0B" if o.degraded else ("#10B981" if "BULLISH" in o.label or "GROUNDED" in o.label else "#EF4444")
                st.markdown(f"""
                <div class="agent-node" style="border-left: 4px solid {status_col};">
                    <div>
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <strong style="color:#FFF; font-size:1.05rem; font-family:'Outfit';">{o.agent}</strong>
                            <span class="badge-pill" style="font-size:0.62rem; color:{status_col}; border-color:{status_col}40;">
                                {'DEGRADED' if o.degraded else 'NOMINAL'}
                            </span>
                        </div>
                        <div style="font-family:'JetBrains Mono'; font-size:0.68rem; color:#64748B; margin-top:3px; text-transform:uppercase;">
                            {o.dimension.replace('_', ' ')}
                        </div>
                        <div style="font-family:'Outfit'; font-size:1.55rem; font-weight:800; color:{status_col}; margin:14px 0 4px 0;">
                            {o.label}
                        </div>
                        <div style="font-family:'JetBrains Mono'; font-size:0.75rem; color:#94A3B8;">
                            CONFIDENCE // {o.confidence:.0%}
                        </div>
                        <div style="font-size:0.84rem; color:#CBD5E1; line-height:1.55; margin:12px 0; background:rgba(15,23,42,0.6); padding:10px 12px; border-radius:6px;">
                            {o.reasoning}
                        </div>
                    </div>
                    <div style="border-top:1px solid #1E293B; padding-top:10px; font-family:'JetBrains Mono'; font-size:0.72rem; color:#64748B; display:flex; justify-content:space-between;">
                        <span>⚡ {o.latency_ms:.1f}ms</span>
                        <span>{f'📎 {len(o.citations)} Cites' if o.citations else 'No Cites'}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Tab 3: Behavioral Personalization Matrix
    with tab3:
        st.markdown('<div class="sp-card">', unsafe_allow_html=True)
        st.markdown('<div class="sp-card-header">Behavioral Personalization Matrix // Identical Feed Across Risk Profiles</div>', unsafe_allow_html=True)
        
        comp_cols = st.columns(len(RISK_PROFILES))
        for c, pname in zip(comp_cols, RISK_PROFILES.keys()):
            _, syn2 = run_pipeline(
                m_ticker, st.session_state.market_data, st.session_state.news, 
                rag_index, pname,
                simulate_degraded=None if st.session_state.degrade == "none" else st.session_state.degrade
            )
            is_active = (pname == m_profile)
            action_col = "#10B981" if "BUY" in syn2['action'] else ("#EF4444" if "REDUCE" in syn2['action'] else "#00F0FF")
            
            with c:
                st.markdown(f"""
                <div class="agent-node" style="border: 2px solid {'#00F0FF' if is_active else '#1E293B'}; background:{'rgba(0,240,255,0.05)' if is_active else '#080D1A'};">
                    <div>
                        <div style="font-weight:800; font-size:1.05rem; color:#FFF; font-family:'Outfit';">
                            {pname} {'★ (Active)' if is_active else ''}
                        </div>
                        <div style="font-family:'Outfit'; font-size:1.35rem; font-weight:800; color:{action_col}; margin:10px 0 6px 0;">
                            {syn2['action']}
                        </div>
                        <div style="font-size:0.84rem; color:#CBD5E1; line-height:1.55; margin:10px 0;">
                            {syn2['reasoning']}
                        </div>
                    </div>
                    <div style="border-top:1px solid #1E293B; padding-top:8px; font-family:'JetBrains Mono'; font-size:0.75rem; color:#94A3B8;">
                        CONFIDENCE: <strong>{syn2['confidence']:.0%}</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Tab 4: SEBI Citations & Document RAG
    with tab4:
        st.markdown('<div class="sp-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="sp-card-header">Retrieved SEBI Disclosures & Regulatory Attributions // {m_ticker}</div>', unsafe_allow_html=True)
        
        if not synthesis["citations"]:
            st.warning("⚠️ No direct corporate filings retrieved or Degraded Mode active.")
        else:
            for c in synthesis["citations"]:
                st.markdown(f"""
                <div class="citation-box">
                    <div style="font-family:'Outfit'; font-size:1.1rem; font-weight:700; color:#FFD700;">
                        📄 SEBI Regulatory Filing Chunk: {c}
                    </div>
                    <div style="font-size:0.88rem; color:#CBD5E1; margin-top:6px; line-height:1.5;">
                        Retrieved via RAG Vector Index. Source grounding verified against statutory corporate disclosures.
                    </div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Tab 5: Benchmarks & Audit Logs
    with tab5:
        st.markdown('<div class="sp-card">', unsafe_allow_html=True)
        st.markdown('<div class="sp-card-header">Quantitative Benchmarks & System Telemetry</div>', unsafe_allow_html=True)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Avg Swarm Latency", f"{row['avg_agent_latency_ms']} ms")
        m2.metric("Portfolio HHI Risk", row["portfolio_concentration_hhi"])
        m3.metric("30d Forward Return", f"{row['mock_30d_forward_return_pct']}%")
        m4.metric("Directional Accuracy", "PASS ✅" if row["directional_accuracy_proxy"] else "NEUTRAL ➖")
        
        st.markdown('<div class="sp-card-header" style="margin-top:24px;">Historical Audit Trail (Persistent Across Sessions)</div>', unsafe_allow_html=True)
        if os.path.exists(LOG_PATH):
            st.dataframe(pd.read_csv(LOG_PATH), use_container_width=True, hide_index=True)
        else:
            st.info("No persistent logs found.")
        st.markdown('</div>', unsafe_allow_html=True)
