import os
import random
import pandas as pd
import streamlit as st

from pipeline import RAGIndex, RISK_PROFILES, LOG_PATH, run_pipeline, log_session
from synthetic_data import TICKERS, generate_market_data, generate_news, generate_filing_corpus, generate_portfolio

# ---------------------------------------------------------------------------
# Page Configuration & Metadata
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SPIDER-SENSE // Tactical Financial Swarm",
    page_icon="🕷️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------------------------
# Ultra-Polished Tactical Cyberpunk Glassmorphism Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600;800&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

/* Global Atmosphere */
* { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: #F8FAFC;
}
.stApp {
    background-color: #030712;
    background-image: 
        radial-gradient(circle at 10% 0%, rgba(230, 36, 41, 0.12) 0%, transparent 40%),
        radial-gradient(circle at 90% 10%, rgba(0, 240, 255, 0.10) 0%, transparent 40%),
        radial-gradient(circle at 50% 100%, rgba(255, 184, 0, 0.05) 0%, transparent 50%),
        radial-gradient(#1E293B 1px, transparent 1px);
    background-size: 100% 100%, 100% 100%, 100% 100%, 24px 24px;
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 3.5rem;
    max-width: 1380px;
}

/* Tactical Hero Banner */
.tactical-hero {
    position: relative;
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(10, 15, 30, 0.98) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-top: 3px solid #E62429;
    border-radius: 16px;
    padding: 22px 28px;
    margin-bottom: 22px;
    box-shadow: 0 20px 40px -15px rgba(0,0,0,0.8), inset 0 1px 0 rgba(255,255,255,0.1);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 16px;
}
.hero-title-group h1 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.02em;
    background: linear-gradient(90deg, #FFFFFF 20%, #00F0FF 80%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-subtitle {
    color: #94A3B8;
    font-size: 0.9rem;
    margin-top: 4px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.hero-badges {
    display: flex;
    gap: 8px;
    align-items: center;
}
.badge-chip {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 800;
    padding: 6px 12px;
    border-radius: 6px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    border: 1px solid rgba(255,255,255,0.1);
    background: #0B1120;
    color: #94A3B8;
}
.badge-chip.pulse {
    border-color: rgba(0, 240, 255, 0.4);
    background: rgba(0, 240, 255, 0.08);
    color: #00F0FF;
    display: flex;
    align-items: center;
    gap: 6px;
}
.pulse-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #00F0FF;
    box-shadow: 0 0 8px #00F0FF;
}

/* Sub-headers */
.hud-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #00F0FF;
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 24px 0 12px 0;
}
.hud-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(0,240,255,0.4), transparent);
}

/* Agent Swarm Card */
.agent-node-card {
    background: #090E1A;
    border: 1px solid #1E293B;
    border-radius: 12px;
    padding: 18px;
    height: 100%;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    transition: all 0.2s ease-in-out;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
.agent-node-card:hover {
    border-color: rgba(0, 240, 255, 0.4);
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(0, 240, 255, 0.1);
}
.node-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 8px;
}
.node-name {
    font-weight: 700;
    font-size: 0.95rem;
    color: #FFF;
}
.node-dim {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #64748B;
    text-transform: uppercase;
    margin-top: 2px;
}
.node-status-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    font-weight: 800;
    padding: 3px 8px;
    border-radius: 4px;
}
.node-signal {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.35rem;
    font-weight: 800;
    margin: 8px 0 4px 0;
}
.node-conf {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #94A3B8;
}
.node-reasoning {
    font-size: 0.82rem;
    color: #CBD5E1;
    line-height: 1.5;
    margin: 12px 0;
    background: rgba(15, 23, 42, 0.7);
    padding: 10px 12px;
    border-radius: 6px;
    border-left: 2px solid #334155;
    flex-grow: 1;
}
.node-footer {
    border-top: 1px solid #1E293B;
    padding-top: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #64748B;
}

/* Master Decision Panel */
.master-verdict-box {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(8, 12, 22, 0.98) 100%);
    border-radius: 14px;
    padding: 24px 28px;
    border: 1px solid #334155;
    box-shadow: 0 12px 30px rgba(0,0,0,0.6);
    margin: 12px 0 20px 0;
}
.verdict-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}
.verdict-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 800;
    letter-spacing: 0.08em;
}
.verdict-action {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.1rem;
    font-weight: 800;
    letter-spacing: -0.02em;
}
.verdict-reasoning {
    font-size: 0.95rem;
    color: #E2E8F0;
    line-height: 1.6;
    margin-top: 10px;
}
.citations-wrapper {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 14px;
    align-items: center;
}
.citation-pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    background: rgba(0, 240, 255, 0.08);
    border: 1px solid rgba(0, 240, 255, 0.25);
    color: #38BDF8;
    padding: 3px 8px;
    border-radius: 4px;
}

/* Side-by-Side Profile Persona Cards */
.persona-card {
    background: #090E1A;
    border: 1px solid #1E293B;
    border-radius: 10px;
    padding: 14px;
    text-align: center;
    transition: all 0.2s ease;
}
.persona-card.active-persona {
    border-color: #00F0FF;
    background: rgba(0, 240, 255, 0.05);
    box-shadow: 0 0 16px rgba(0, 240, 255, 0.15);
}
.persona-name {
    font-weight: 700;
    font-size: 0.85rem;
    color: #F8FAFC;
    text-transform: capitalize;
}
.persona-action {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.95rem;
    font-weight: 700;
    margin-top: 6px;
}

/* Metric KPI HUD Cards */
div[data-testid="stMetric"] {
    background: #090E1A;
    border: 1px solid #1E293B;
    border-radius: 10px;
    padding: 12px 18px;
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
    font-size: 1.5rem !important;
    font-weight: 800 !important;
    color: #F8FAFC !important;
}

/* Interactive Dispatch Button */
div.stButton > button {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    background: linear-gradient(90deg, #E62429 0%, #FF4655 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.75rem 2rem !important;
    box-shadow: 0 4px 18px rgba(230, 36, 41, 0.4) !important;
    transition: all 0.2s ease !important;
    width: 100%;
}
div.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(230, 36, 41, 0.6) !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# UI Helpers & Visual Color Matrix
# ---------------------------------------------------------------------------
def get_node_theme(o):
    """Dynamically calculates tactical borders and status pills."""
    if o.degraded:
        return {
            "accent": "#F59E0B",
            "bg": "rgba(245, 158, 11, 0.12)",
            "text": "#F59E0B",
            "status": "FALLBACK ACTIVE"
        }
    
    bullish = {"BULLISH", "POSITIVE", "VOLUME_SPIKE", "GROUNDED"}
    bearish = {"BEARISH", "NEGATIVE", "HIGH_RISK"}
    
    if o.label in bullish:
        return {
            "accent": "#10B981",
            "bg": "rgba(16, 185, 129, 0.12)",
            "text": "#10B981",
            "status": "NOMINAL"
        }
    if o.label in bearish:
        return {
            "accent": "#EF4444",
            "bg": "rgba(239, 68, 68, 0.12)",
            "text": "#EF4444",
            "status": "ALERT"
        }
    return {
        "accent": "#00F0FF",
        "bg": "rgba(0, 240, 255, 0.12)",
        "text": "#00F0FF",
        "status": "NEUTRAL"
    }


def render_node_ui(o):
    theme = get_node_theme(o)
    cites = f"<span>📎 {', '.join(o.citations)}</span>" if o.citations else ""

    html = f"""
    <div class="agent-node-card" style="border-left: 3px solid {theme['accent']};">
        <div>
            <div class="node-top">
                <div>
                    <div class="node-name">{o.agent}</div>
                    <div class="node-dim">{o.dimension.replace('_', ' ')}</div>
                </div>
                <span class="node-status-tag" style="background:{theme['bg']}; color:{theme['text']}; border: 1px solid {theme['accent']}40;">
                    {theme['status']}
                </span>
            </div>
            <div class="node-signal" style="color:{theme['accent']};">{o.label}</div>
            <div class="node-conf">CONFIDENCE // {o.confidence:.0%}</div>
            <div class="node-reasoning">{o.reasoning}</div>
        </div>
        <div class="node-footer">
            <span>⚡ {o.latency_ms:.1f}ms</span>
            {cites}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_master_verdict(synthesis):
    if synthesis["conflict"]:
        border_col, text_col = "#F59E0B", "#FBBF24"
        badge_text = "⚠️ CONFLICT DETECTED & RESOLVED"
    elif synthesis["action"].startswith("CONSIDER BUY"):
        border_col, text_col = "#10B981", "#34D399"
        badge_text = "🟢 HIGH-CONVICTION ALIGNMENT"
    elif synthesis["action"].startswith("CONSIDER REDUCE"):
        border_col, text_col = "#EF4444", "#F87171"
        badge_text = "🔴 RISK MITIGATION / CAPITAL PRESERVATION"
    else:
        border_col, text_col = "#00F0FF", "#38BDF8"
        badge_text = "🔵 DEFENSIVE / BALANCED ALLOCATION"

    cites_html = ""
    if synthesis["citations"]:
        chips = "".join([f'<span class="citation-pill">SEBI // {c}</span>' for c in synthesis["citations"]])
        cites_html = f'<div class="citations-wrapper"><span style="font-size:0.75rem; color:#64748B; text-transform:uppercase; font-weight:700;">Attributions:</span> {chips}</div>'

    degraded_notice = ""
    if synthesis["degraded_agents"]:
        degraded_notice = f"""
        <div style="margin-top:14px; padding:10px 14px; border-radius:6px; background:rgba(245,158,11,0.08); border:1px dashed #F59E0B; font-size:0.8rem; color:#FCD34D;">
            ⚠️ <strong>Degraded Feed Fallback:</strong> Swarm bypassed offline telemetry on: {", ".join(synthesis['degraded_agents'])}. Confidence scores penalized to prevent ungrounded algorithmic execution.
        </div>
        """

    html = f"""
    <div class="master-verdict-box" style="border-left: 4px solid {border_col};">
        <div class="verdict-header">
            <span class="verdict-tag" style="color:{text_col};">{badge_text}</span>
            <span style="font-family:'JetBrains Mono'; font-size:0.78rem; color:#94A3B8;">
                SYNTHESIS CONFIDENCE: <strong style="color:#FFF;">{synthesis['confidence']:.0%}</strong>
            </span>
        </div>
        <div class="verdict-action" style="color:{text_col};">{synthesis['action']}</div>
        <div class="verdict-reasoning">{synthesis['reasoning']}</div>
        {cites_html}
        {degraded_notice}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
    st.progress(synthesis["confidence"])


# ---------------------------------------------------------------------------
# Hero Section
# ---------------------------------------------------------------------------
st.markdown("""
<div class="tactical-hero">
    <div class="hero-title-group">
        <h1>🕷️ SPIDER-SENSE // Autonomous Swarm</h1>
        <div class="hero-subtitle">
            <span>Explainable Multi-Agent Financial Intelligence for Retail Investors</span>
        </div>
    </div>
    <div class="hero-badges">
        <span class="badge-chip pulse"><span class="pulse-dot"></span>Swarm Online</span>
        <span class="badge-chip">PS-01 // Sprint 1</span>
        <span class="badge-chip" style="border-color: rgba(230,36,41,0.5); color:#FF4655;">Team YOLOTECH</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Engine Initialization & State
# ---------------------------------------------------------------------------
@st.cache_resource
def get_rag_index():
    return RAGIndex(generate_filing_corpus())

rag_index = get_rag_index()
portfolio = generate_portfolio()

# ---------------------------------------------------------------------------
# Control Parameters & Inputs
# ---------------------------------------------------------------------------
with st.container():
    st.markdown('<div class="hud-title">Target Parameters & Telemetry Configuration</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1.2])
    with c1:
        ticker = st.selectbox("Target Asset", TICKERS, help="NSE listed equities universe")
    with c2:
        profile_name = st.selectbox("Investor Persona", list(RISK_PROFILES.keys()), index=1)
    with c3:
        scenario = st.selectbox("Market Scenario", ["normal", "crash", "positive_news", "negative_news"])
    with c4:
        degrade = st.selectbox("Fault-Injection (Degraded Mode)", ["none", "momentum", "volume", "sentiment"])

    st.write("")
    run = st.button("🚀 DISPATCH MULTI-AGENT SWARM", use_container_width=True)

# ---------------------------------------------------------------------------
# Portfolio Watchlist Section
# ---------------------------------------------------------------------------
with st.expander("💼 User Portfolio Composition & Current Holdings", expanded=False):
    st.dataframe(
        pd.DataFrame(list(portfolio.items()), columns=["Asset", "Holding Valuation (₹)"]),
        use_container_width=True,
        hide_index=True
    )

# ---------------------------------------------------------------------------
# Execution Pipeline & Multi-Agent Swarm
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

    # Parallel Execution Dispatch
    outputs, synthesis = run_pipeline(
        ticker, market_data, news, rag_index, profile_name,
        simulate_degraded=None if degrade == "none" else degrade,
    )

    # 1. Parallel Agent Telemetry
    st.markdown('<div class="hud-title">Parallel Domain Agent Telemetry</div>', unsafe_allow_html=True)
    agent_cols = st.columns(len(outputs))
    for c, o in zip(agent_cols, outputs):
        with c:
            render_node_ui(o)

    # 2. Master Decision Synthesis
    st.markdown('<div class="hud-title">Orchestrated Master Synthesis</div>', unsafe_allow_html=True)
    render_master_verdict(synthesis)

    # 3. Behavioral Personalization Matrix
    st.markdown('<div class="hud-title">Behavioral Personalization Matrix (Identical Signal Across Personas)</div>', unsafe_allow_html=True)
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
            <div class="persona-card {'active-persona' if is_active else ''}">
                <div class="persona-name">{pname} {'★' if is_active else ''}</div>
                <div class="persona-action" style="color:{action_col};">{syn2['action']}</div>
                <div style="font-family:'JetBrains Mono'; font-size:0.72rem; color:#64748B; margin-top:4px;">
                    CONF: {syn2['confidence']:.0%}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # 4. Session Metrics HUD
    forward_return_mock = random.uniform(-8, 8)
    row = log_session(ticker, profile_name, outputs, synthesis, forward_return_mock, portfolio)

    st.markdown('<div class="hud-title">Session Telemetry & Quantitative Metrics</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Avg Swarm Latency", f"{row['avg_agent_latency_ms']} ms")
    m2.metric("Portfolio HHI Risk", row["portfolio_concentration_hhi"])
    m3.metric("30d Forward Return", f"{row['mock_30d_forward_return_pct']}%")
    m4.metric("Directional Accuracy", "PASS ✅" if row["directional_accuracy_proxy"] else "NEUTRAL ➖")

# ---------------------------------------------------------------------------
# Persistent Historical Logs
# ---------------------------------------------------------------------------
st.markdown('<div class="hud-title">Audit Trail & Session Persistence Log</div>', unsafe_allow_html=True)
if os.path.exists(LOG_PATH):
    st.dataframe(pd.read_csv(LOG_PATH), use_container_width=True, hide_index=True)
else:
    st.info("No logs found. Run a multi-agent dispatch above to generate an audit trail.")
