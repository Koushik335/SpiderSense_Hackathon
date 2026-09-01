import os
import random
import pandas as pd
import streamlit as st

from pipeline import RAGIndex, RISK_PROFILES, LOG_PATH, run_pipeline, log_session
from synthetic_data import TICKERS, generate_market_data, generate_news, generate_filing_corpus, generate_portfolio

# ---------------------------------------------------------------------------
# Streamlit Configuration & Metadata
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SPIDER-SENSE // Autonomous Swarm Command",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------------------------
# High-Impact Cyberpunk Comic UI & 3D Chromatic Typography
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bangers&family=Space+Grotesk:wght@500;700;800&family=JetBrains+Mono:wght@400;600;800&family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');

* { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: #F8FAFC;
}

.stApp {
    background-color: #030712;
    background-image: 
        radial-gradient(circle at 10% 0%, rgba(230, 36, 41, 0.16) 0%, transparent 45%),
        radial-gradient(circle at 90% 10%, rgba(0, 240, 255, 0.14) 0%, transparent 45%),
        radial-gradient(circle at 50% 100%, rgba(255, 184, 0, 0.08) 0%, transparent 50%),
        radial-gradient(#1E293B 1.2px, transparent 1.2px);
    background-size: 100% 100%, 100% 100%, 100% 100%, 26px 26px;
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 3.5rem;
    max-width: 1400px;
}

/* 3D Comic Chromatic Typography */
.comic-logo-text {
    font-family: 'Bangers', cursive !important;
    font-size: 3rem;
    font-style: italic;
    letter-spacing: 2px;
    color: #FFFFFF;
    text-transform: uppercase;
    margin: 0;
    line-height: 1.05;
    text-shadow: 
        -3px -3px 0 #00F0FF,
        3px 3px 0 #E62429,
        6px 6px 0 #000000;
}

.comic-subhead {
    font-family: 'Bangers', cursive !important;
    font-size: 1.6rem;
    font-style: italic;
    letter-spacing: 1.5px;
    color: #FFFFFF;
    text-shadow: 
        -2px -2px 0 #00F0FF,
        2px 2px 0 #E62429,
        4px 4px 0 #000000;
}

/* Tactical Hero Hub */
.tactical-hero {
    position: relative;
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.96) 0%, rgba(8, 12, 24, 0.98) 100%);
    border: 2px solid #1E293B;
    border-top: 4px solid #E62429;
    border-radius: 18px;
    padding: 24px 32px;
    margin-bottom: 24px;
    box-shadow: 0 20px 45px -10px rgba(0,0,0,0.9), inset 0 1px 0 rgba(255,255,255,0.12);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 16px;
}
.hero-desc {
    color: #94A3B8;
    font-size: 0.92rem;
    margin-top: 6px;
    font-weight: 500;
}
.hero-badges {
    display: flex;
    gap: 10px;
    align-items: center;
    flex-wrap: wrap;
}
.badge-chip {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 800;
    padding: 6px 14px;
    border-radius: 6px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border: 1px solid rgba(255,255,255,0.12);
    background: #0B1120;
    color: #94A3B8;
}
.badge-chip.pulse {
    border-color: rgba(0, 240, 255, 0.5);
    background: rgba(0, 240, 255, 0.1);
    color: #00F0FF;
    display: flex;
    align-items: center;
    gap: 8px;
    box-shadow: 0 0 14px rgba(0, 240, 255, 0.2);
}
.pulse-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #00F0FF;
    box-shadow: 0 0 10px #00F0FF;
}

/* Control Hub Header */
.hud-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.88rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: #00F0FF;
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 26px 0 14px 0;
}
.hud-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(0,240,255,0.5), transparent);
}

/* Agent Swarm Node Cards */
.agent-node-card {
    background: #090E1A;
    border: 1px solid #1E293B;
    border-radius: 14px;
    padding: 20px;
    height: 100%;
    box-shadow: 0 8px 24px rgba(0,0,0,0.5);
    transition: all 0.25s ease-in-out;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
.agent-node-card:hover {
    border-color: rgba(0, 240, 255, 0.5);
    transform: translateY(-3px);
    box-shadow: 0 12px 35px rgba(0, 240, 255, 0.15);
}
.node-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 10px;
}
.node-name {
    font-weight: 800;
    font-size: 1.02rem;
    color: #FFFFFF;
}
.node-dim {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-top: 3px;
}
.node-status-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 800;
    padding: 3px 9px;
    border-radius: 5px;
    letter-spacing: 0.05em;
}
.node-signal {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.45rem;
    font-weight: 800;
    margin: 10px 0 4px 0;
    letter-spacing: -0.01em;
}
.node-conf {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #94A3B8;
}
.node-reasoning {
    font-size: 0.84rem;
    color: #CBD5E1;
    line-height: 1.55;
    margin: 14px 0;
    background: rgba(15, 23, 42, 0.75);
    padding: 12px 14px;
    border-radius: 8px;
    border-left: 3px solid #334155;
    flex-grow: 1;
}
.node-footer {
    border-top: 1px solid #1E293B;
    padding-top: 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #64748B;
}

/* Master Decision Hub */
.master-verdict-box {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.97) 0%, rgba(6, 10, 20, 0.99) 100%);
    border-radius: 16px;
    padding: 26px 32px;
    border: 1px solid #334155;
    box-shadow: 0 16px 35px rgba(0,0,0,0.7);
    margin: 14px 0 24px 0;
}
.verdict-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}
.verdict-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 0.08em;
}
.verdict-action {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.3rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1.2;
}
.verdict-reasoning {
    font-size: 0.98rem;
    color: #E2E8F0;
    line-height: 1.65;
    margin-top: 12px;
}
.citations-wrapper {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 16px;
    align-items: center;
}
.citation-pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.74rem;
    background: rgba(0, 240, 255, 0.08);
    border: 1px solid rgba(0, 240, 255, 0.3);
    color: #38BDF8;
    padding: 4px 10px;
    border-radius: 5px;
    font-weight: 600;
}

/* Persona Testing Matrix */
.persona-card {
    background: #090E1A;
    border: 1px solid #1E293B;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    transition: all 0.2s ease;
}
.persona-card.active-persona {
    border-color: #00F0FF;
    background: rgba(0, 240, 255, 0.06);
    box-shadow: 0 0 20px rgba(0, 240, 255, 0.18);
}
.persona-name {
    font-weight: 800;
    font-size: 0.9rem;
    color: #F8FAFC;
    text-transform: capitalize;
}
.persona-action {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.05rem;
    font-weight: 800;
    margin-top: 6px;
}

/* Metric KPI HUD Cards */
div[data-testid="stMetric"] {
    background: #090E1A;
    border: 1px solid #1E293B;
    border-radius: 12px;
    padding: 14px 20px;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.06);
}
div[data-testid="stMetric"] label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.74rem !important;
    color: #94A3B8 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.6rem !important;
    font-weight: 800 !important;
    color: #F8FAFC !important;
}

/* Custom Dispatch Action Button */
div.stButton > button {
    font-family: 'Bangers', cursive !important;
    font-style: italic !important;
    font-size: 1.55rem !important;
    letter-spacing: 2px !important;
    background: linear-gradient(90deg, #E62429 0%, #FF4655 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.85rem 2.2rem !important;
    box-shadow: 0 4px 20px rgba(230, 36, 41, 0.45) !important;
    transition: all 0.2s ease !important;
    width: 100%;
}
div.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(230, 36, 41, 0.7) !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# UI Helpers & Visual Logic
# ---------------------------------------------------------------------------
def get_node_theme(o):
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
    <div class="agent-node-card" style="border-left: 3.5px solid {theme['accent']};">
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
        cites_html = f'<div class="citations-wrapper"><span style="font-size:0.75rem; color:#64748B; text-transform:uppercase; font-weight:800;">Attributions:</span> {chips}</div>'

    degraded_notice = ""
    if synthesis["degraded_agents"]:
        degraded_notice = f"""
        <div style="margin-top:14px; padding:10px 14px; border-radius:8px; background:rgba(245,158,11,0.08); border:1px dashed #F59E0B; font-size:0.82rem; color:#FCD34D;">
            ⚠️ <strong>Degraded Feed Fallback:</strong> Swarm bypassed offline telemetry on: {", ".join(synthesis['degraded_agents'])}. Confidence scores penalized to prevent ungrounded algorithmic execution.
        </div>
        """

    html = f"""
    <div class="master-verdict-box" style="border-left: 4.5px solid {border_col};">
        <div class="verdict-header">
            <span class="verdict-tag" style="color:{text_col};">{badge_text}</span>
            <span style="font-family:'JetBrains Mono'; font-size:0.8rem; color:#94A3B8;">
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
# Hero Header Banner
# ---------------------------------------------------------------------------
st.markdown("""
<div class="tactical-hero">
    <div>
        <div class="comic-logo-text">🕷️ SPIDER-SENSE FINANCIAL</div>
        <div class="hero-desc">
            Autonomous Multi-Agent Retail Intelligence Network // Grounded Reasoning
        </div>
    </div>
    <div class="hero-badges">
        <span class="badge-chip pulse"><span class="pulse-dot"></span>Swarm Active</span>
        <span class="badge-chip">PS-01 // Sprint 1</span>
        <span class="badge-chip" style="border-color: rgba(230,36,41,0.6); color:#FF4655;">Team YOLOTECH</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# RAG Corpus & State
# ---------------------------------------------------------------------------
@st.cache_resource
def get_rag_index():
    return RAGIndex(generate_filing_corpus())

rag_index = get_rag_index()
portfolio = generate_portfolio()

# ---------------------------------------------------------------------------
# Telemetry Parameters & Configuration Deck
# ---------------------------------------------------------------------------
with st.container():
    st.markdown('<div class="hud-title">Telemetry Configuration & Target Parameters</div>', unsafe_allow_html=True)
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
    run = st.button("DISPATCH MULTI-AGENT SWARM", use_container_width=True)

# ---------------------------------------------------------------------------
# Portfolio State
# ---------------------------------------------------------------------------
with st.expander("💼 User Portfolio Composition & Current Holdings", expanded=False):
    st.dataframe(
        pd.DataFrame(list(portfolio.items()), columns=["Asset", "Holding Valuation (₹)"]),
        use_container_width=True,
        hide_index=True
    )

# ---------------------------------------------------------------------------
# Swarm Execution & Analytics
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

    # Parallel Pipeline Execution
    outputs, synthesis = run_pipeline(
        ticker, market_data, news, rag_index, profile_name,
        simulate_degraded=None if degrade == "none" else degrade,
    )

    # 1. Parallel Telemetry Grid
    st.markdown('<div class="hud-title">Parallel Domain Agent Telemetry</div>', unsafe_allow_html=True)
    agent_cols = st.columns(len(outputs))
    for c, o in zip(agent_cols, outputs):
        with c:
            render_node_ui(o)

    # 2. Master Decision Synthesis
    st.markdown('<div class="hud-title">Orchestrated Master Synthesis</div>', unsafe_allow_html=True)
    render_master_verdict(synthesis)

    # 3. Behavioral Personalization Matrix
    st.markdown('<div class="hud-title">Behavioral Personalization Matrix (Identical Signals Across Personas)</div>', unsafe_allow_html=True)
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
                <div style="font-family:'JetBrains Mono'; font-size:0.75rem; color:#64748B; margin-top:4px;">
                    CONF: {syn2['confidence']:.0%}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # 4. Telemetry Metrics
    forward_return_mock = random.uniform(-8, 8)
    row = log_session(ticker, profile_name, outputs, synthesis, forward_return_mock, portfolio)

    st.markdown('<div class="hud-title">Session Telemetry & Quantitative Metrics</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Avg Swarm Latency", f"{row['avg_agent_latency_ms']} ms")
    m2.metric("Portfolio HHI Risk", row["portfolio_concentration_hhi"])
    m3.metric("30d Forward Return", f"{row['mock_30d_forward_return_pct']}%")
    m4.metric("Directional Accuracy", "PASS ✅" if row["directional_accuracy_proxy"] else "NEUTRAL ➖")

# ---------------------------------------------------------------------------
# Audit Logs
# ---------------------------------------------------------------------------
st.markdown('<div class="hud-title">Audit Trail & Session Persistence Log</div>', unsafe_allow_html=True)
if os.path.exists(LOG_PATH):
    st.dataframe(pd.read_csv(LOG_PATH), use_container_width=True, hide_index=True)
else:
    st.info("No logs found. Run a multi-agent dispatch above to generate an audit trail.")
