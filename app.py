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
    page_title="SPIDER-SENSE // Autonomous Financial Intelligence",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------------------------
# High-Impact Terminal / Glassmorphic CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

/* Global Reset */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #F1F5F9;
}
.stApp {
    background-color: #060913;
    background-image: 
        radial-gradient(at 0% 0%, rgba(230, 36, 41, 0.08) 0px, transparent 50%),
        radial-gradient(at 100% 0%, rgba(0, 240, 255, 0.08) 0px, transparent 50%),
        radial-gradient(#1E293B 1px, transparent 1px);
    background-size: 100% 100%, 100% 100%, 28px 28px;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3.5rem;
    max-width: 1320px;
}

/* Header / Hero */
.hero-container {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(10, 15, 29, 0.95) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-top: 3px solid #E62429;
    border-radius: 14px;
    padding: 24px 30px;
    margin-bottom: 24px;
    box-shadow: 0 12px 30px -10px rgba(0,0,0,0.8);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 16px;
}
.hero-title-group h1 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 2.1rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.02em;
    background: linear-gradient(90deg, #FFFFFF 30%, #00F0FF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-title-group p {
    color: #94A3B8;
    margin: 4px 0 0 0;
    font-size: 0.92rem;
}
.badge-group {
    display: flex;
    gap: 8px;
    align-items: center;
}
.hud-pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 5px 12px;
    border-radius: 6px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border: 1px solid rgba(255,255,255,0.1);
    background: #0B1120;
    color: #94A3B8;
}
.hud-pill.live {
    border-color: rgba(0, 240, 255, 0.4);
    background: rgba(0, 240, 255, 0.08);
    color: #00F0FF;
}

/* Control Hub Container */
.control-panel {
    background: rgba(13, 19, 34, 0.85);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 20px;
}

/* Section Titles */
.hud-heading {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.88rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #00F0FF;
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 24px 0 12px 0;
}
.hud-heading::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(0,240,255,0.3), transparent);
}

/* Agent Card Design */
.agent-grid-card {
    background: #090E1A;
    border: 1px solid #1E293B;
    border-radius: 12px;
    padding: 18px;
    height: 100%;
    position: relative;
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.agent-grid-card:hover {
    transform: translateY(-2px);
    border-color: rgba(0, 240, 255, 0.3);
}
.agent-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 12px;
}
.agent-name {
    font-weight: 700;
    font-size: 0.95rem;
    color: #F8FAFC;
}
.agent-dim {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-top: 2px;
}
.agent-status-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 700;
    padding: 2px 7px;
    border-radius: 4px;
}
.agent-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.35rem;
    font-weight: 700;
    margin: 8px 0 4px 0;
}
.agent-conf {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #94A3B8;
}
.agent-reasoning {
    font-size: 0.82rem;
    color: #CBD5E1;
    line-height: 1.5;
    margin: 12px 0;
    background: rgba(15, 23, 42, 0.6);
    padding: 10px;
    border-radius: 6px;
    border-left: 2px solid #334155;
}
.agent-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-top: 1px solid #1E293B;
    padding-top: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #64748B;
}

/* Master Decision HUD */
.master-decision-card {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(8, 12, 22, 0.98) 100%);
    border-radius: 14px;
    padding: 24px 28px;
    border: 1px solid #334155;
    box-shadow: 0 8px 24px rgba(0,0,0,0.5);
    margin: 12px 0 20px 0;
    position: relative;
    overflow: hidden;
}
.master-action {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.01em;
}
.master-reasoning {
    font-size: 0.95rem;
    color: #E2E8F0;
    line-height: 1.6;
    margin-top: 10px;
}
.citation-chip-container {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 14px;
    align-items: center;
}
.citation-chip {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    background: rgba(0, 240, 255, 0.08);
    border: 1px solid rgba(0, 240, 255, 0.25);
    color: #38BDF8;
    padding: 3px 8px;
    border-radius: 4px;
}

/* Profile Comparison Chips */
.profile-compare-card {
    background: #090E1A;
    border: 1px solid #1E293B;
    border-radius: 10px;
    padding: 14px;
    text-align: center;
    transition: all 0.2s;
}
.profile-compare-card.active {
    border-color: #00F0FF;
    background: rgba(0, 240, 255, 0.04);
}
.profile-title {
    font-weight: 700;
    font-size: 0.85rem;
    color: #F8FAFC;
    text-transform: capitalize;
}
.profile-action {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.95rem;
    font-weight: 700;
    margin-top: 6px;
}

/* Metric KPI HUD */
div[data-testid="stMetric"] {
    background: #090E1A;
    border: 1px solid #1E293B;
    border-radius: 10px;
    padding: 12px 16px;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);
}
div[data-testid="stMetric"] label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    color: #94A3B8 !important;
    text-transform: uppercase;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.45rem !important;
    font-weight: 700 !important;
    color: #F8FAFC !important;
}

/* Custom Streamlit Button Styling */
div.stButton > button {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    background: linear-gradient(90deg, #E62429 0%, #FF4655 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.65rem 1.8rem !important;
    box-shadow: 0 4px 14px rgba(230, 36, 41, 0.4) !important;
    transition: all 0.2s ease !important;
    width: 100%;
}
div.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(230, 36, 41, 0.6) !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# UI Helpers & Renderers
# ---------------------------------------------------------------------------
def get_agent_colors(o):
    """Returns dynamic color sets for specialized agent cards."""
    if o.degraded:
        return {"accent": "#F59E0B", "bg": "rgba(245,158,11,0.12)", "text": "#F59E0B", "status": "DEGRADED"}
    
    bullish = {"BULLISH", "POSITIVE", "VOLUME_SPIKE", "GROUNDED"}
    bearish = {"BEARISH", "NEGATIVE", "HIGH_RISK"}
    
    if o.label in bullish:
        return {"accent": "#10B981", "bg": "rgba(16,185,129,0.12)", "text": "#10B981", "status": "NOMINAL"}
    if o.label in bearish:
        return {"accent": "#EF4444", "bg": "rgba(239,68,68,0.12)", "text": "#EF4444", "status": "ALERT"}
    
    return {"accent": "#00F0FF", "bg": "rgba(0,240,255,0.12)", "text": "#00F0FF", "status": "NEUTRAL"}


def render_agent_card_ui(o):
    cfg = get_agent_colors(o)
    cites = ""
    if o.citations:
        cites = f"<span>📎 {', '.join(o.citations)}</span>"

    html = f"""
    <div class="agent-grid-card" style="border-left: 3px solid {cfg['accent']};">
        <div class="agent-header">
            <div>
                <div class="agent-name">{o.agent}</div>
                <div class="agent-dim">{o.dimension.replace('_', ' ')}</div>
            </div>
            <span class="agent-status-badge" style="background:{cfg['bg']}; color:{cfg['text']}; border: 1px solid {cfg['accent']}40;">
                {cfg['status']}
            </span>
        </div>
        <div class="agent-label" style="color:{cfg['accent']};">{o.label}</div>
        <div class="agent-conf">CONFIDENCE // {o.confidence:.0%}</div>
        <div class="agent-reasoning">{o.reasoning}</div>
        <div class="agent-footer">
            <span>⚡ {o.latency_ms:.1f}ms</span>
            {cites}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_decision_hud(synthesis):
    if synthesis["conflict"]:
        border_col, text_col = "#F59E0B", "#FBBF24"
        badge_text = "⚠️ CONFLICT DETECTED & RESOLVED"
    elif synthesis["action"].startswith("CONSIDER BUY"):
        border_col, text_col = "#10B981", "#34D399"
        badge_text = "🟢 HIGH-CONVICTION OPPORTUNITY"
    elif synthesis["action"].startswith("CONSIDER REDUCE"):
        border_col, text_col = "#EF4444", "#F87171"
        badge_text = "🔴 RISK MITIGATION / EXIT"
    else:
        border_col, text_col = "#00F0FF", "#38BDF8"
        badge_text = "🔵 NEUTRAL ALLOCATION"

    cites_html = ""
    if synthesis["citations"]:
        chips = "".join([f'<span class="citation-chip">SEBI // {c}</span>' for c in synthesis["citations"]])
        cites_html = f'<div class="citation-chip-container"><span style="font-size:0.75rem; color:#64748B; text-transform:uppercase;">Attributions:</span> {chips}</div>'

    degraded_notice = ""
    if synthesis["degraded_agents"]:
        degraded_notice = f"""
        <div style="margin-top:12px; padding:8px 12px; border-radius:6px; background:rgba(245,158,11,0.08); border:1px dashed #F59E0B; font-size:0.78rem; color:#FCD34D;">
            ⚠️ <strong>Degraded Telemetry Alert:</strong> Fallback routing active for: {", ".join(synthesis['degraded_agents'])}. Confidence scores penalized to prevent ungrounded algorithmic execution.
        </div>
        """

    html = f"""
    <div class="master-decision-card" style="border-left: 4px solid {border_col};">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
            <span style="font-family:'JetBrains Mono'; font-size:0.72rem; color:{text_col}; letter-spacing:0.08em; font-weight:700;">
                {badge_text}
            </span>
            <span style="font-family:'JetBrains Mono'; font-size:0.75rem; color:#94A3B8;">
                SYNTHESIS CONFIDENCE: <strong>{synthesis['confidence']:.0%}</strong>
            </span>
        </div>
        <div class="master-action" style="color:{text_col};">{synthesis['action']}</div>
        <div class="master-reasoning">{synthesis['reasoning']}</div>
        {cites_html}
        {degraded_notice}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
    st.progress(synthesis["confidence"])


# ---------------------------------------------------------------------------
# Header Section
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title-group">
        <h1>🕷️ SPIDER-SENSE // Financial Intelligence</h1>
        <p>Explainable, Multi-Agent Autonomous Reasoning Architecture for Retail Investors</p>
    </div>
    <div class="badge-group">
        <span class="hud-pill live">● Swarm Online</span>
        <span class="hud-pill">PS-01 // Sprint 1</span>
        <span class="hud-pill" style="border-color: rgba(230,36,41,0.5); color:#FF4655;">Team YOLOTECH</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Corpus & State Initialization
# ---------------------------------------------------------------------------
@st.cache_resource
def get_rag_index():
    return RAGIndex(generate_filing_corpus())

rag_index = get_rag_index()
portfolio = generate_portfolio()

# ---------------------------------------------------------------------------
# Controls Panel
# ---------------------------------------------------------------------------
with st.container():
    st.markdown('<div class="hud-heading">Telemetry Configuration & Target Parameters</div>', unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1.2])
    with c1:
        ticker = st.selectbox("Target Asset", TICKERS, help="NSE listed equities")
    with c2:
        profile_name = st.selectbox("Investor Persona", list(RISK_PROFILES.keys()), index=1)
    with c3:
        scenario = st.selectbox("Simulated Market Environment", ["normal", "crash", "positive_news", "negative_news"])
    with c4:
        degrade = st.selectbox("Fault-Injection (Degraded Mode)", ["none", "momentum", "volume", "sentiment"])

    st.write("")
    run = st.button("🚀 DISPATCH MULTI-AGENT SWARM", use_container_width=True)

# ---------------------------------------------------------------------------
# Watchlist / Portfolio Overview
# ---------------------------------------------------------------------------
with st.expander("💼 User Portfolio & Watchlist State", expanded=False):
    st.dataframe(
        pd.DataFrame(list(portfolio.items()), columns=["Asset", "Holding Valuation (₹)"]),
        use_container_width=True,
        hide_index=True
    )

# ---------------------------------------------------------------------------
# Multi-Agent Execution & Rendering
# ---------------------------------------------------------------------------
if run:
    crash = (scenario == "crash")
    market_data = generate_market_data(ticker, crash=crash)
    news_sentiment = (
        "positive" if scenario == "positive_news" 
        else "negative" if scenario == "negative_news" 
        else "mixed"
    )
    news = generate_news(ticker, news_sentiment)

    # Parallel Agent Orchestration
    outputs, synthesis = run_pipeline(
        ticker, market_data, news, rag_index, profile_name,
        simulate_degraded=None if degrade == "none" else degrade,
    )

    # 1. Agent Swarm Results
    st.markdown('<div class="hud-heading">Parallel Agent Telemetry & Reasoning Traces</div>', unsafe_allow_html=True)
    agent_cols = st.columns(len(outputs))
    for c, o in zip(agent_cols, outputs):
        with c:
            render_agent_card_ui(o)

    # 2. Synthesized Master Recommendation
    st.markdown('<div class="hud-heading">Synthesized & Grounded Recommendation</div>', unsafe_allow_html=True)
    render_decision_hud(synthesis)

    # 3. Behavioral Personalization Proof (Same Data -> Across All Risk Profiles)
    st.markdown('<div class="hud-heading">Behavioral Personalization Proof (Identical Input across Personas)</div>', unsafe_allow_html=True)
    comp_cols = st.columns(len(RISK_PROFILES))
    for c, pname in zip(comp_cols, RISK_PROFILES.keys()):
        _, syn2 = run_pipeline(
            ticker, market_data, news, rag_index, pname,
            simulate_degraded=None if degrade == "none" else degrade
        )
        is_active = (pname == profile_name)
        action_col = "#10B981" if "BUY" in syn2['action'] else ("#EF4444" if "REDUCE" in syn2['action'] else "#00F0FF")
        
        with c:
            st.markdown(f"""
            <div class="profile-compare-card {'active' if is_active else ''}">
                <div class="profile-title">{pname} {'★' if is_active else ''}</div>
                <div class="profile-action" style="color:{action_col};">{syn2['action']}</div>
                <div style="font-family:'JetBrains Mono'; font-size:0.72rem; color:#64748B; margin-top:4px;">
                    CONF: {syn2['confidence']:.0%}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # 4. Session Metrics HUD
    forward_return_mock = random.uniform(-8, 8)
    row = log_session(ticker, profile_name, outputs, synthesis, forward_return_mock, portfolio)

    st.markdown('<div class="hud-heading">Session Telemetry & Quantitative Metrics</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Avg Swarm Latency", f"{row['avg_agent_latency_ms']} ms")
    m2.metric("Portfolio HHI Risk Index", row["portfolio_concentration_hhi"])
    m3.metric("30d Forward Return Proxy", f"{row['mock_30d_forward_return_pct']}%")
    m4.metric("Directional Accuracy", "PASS ✅" if row["directional_accuracy_proxy"] else "NEUTRAL ➖")

# ---------------------------------------------------------------------------
# Persistent Historical Logs
# ---------------------------------------------------------------------------
st.markdown('<div class="hud-heading">Audit Trail & Session Persistence Log</div>', unsafe_allow_html=True)
if os.path.exists(LOG_PATH):
    st.dataframe(pd.read_csv(LOG_PATH), use_container_width=True, hide_index=True)
else:
    st.info("No logs found. Run a multi-agent dispatch above to generate an audit trail.")
