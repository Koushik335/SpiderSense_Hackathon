import os
import random

import pandas as pd
import streamlit as st

from pipeline import RAGIndex, RISK_PROFILES, LOG_PATH, run_pipeline, log_session
from synthetic_data import TICKERS, generate_market_data, generate_news, generate_filing_corpus, generate_portfolio

st.set_page_config(
    page_title="Sentinel — Multi-Agent Investment Intelligence",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; }

.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }

/* Hero */
.hero {
    padding: 28px 32px;
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(34,211,166,0.15) 0%, rgba(20,30,45,0.4) 60%);
    border: 1px solid rgba(34,211,166,0.25);
    margin-bottom: 24px;
}
.hero h1 {
    font-size: 2.1rem;
    margin: 0 0 6px 0;
    background: linear-gradient(90deg, #22D3A6, #7DD3FC);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p { color: #9FB0C0; margin: 0; font-size: 0.95rem; }
.hero .tag {
    display: inline-block; margin-top: 10px; padding: 4px 12px;
    border-radius: 999px; background: rgba(34,211,166,0.12);
    border: 1px solid rgba(34,211,166,0.35); color: #22D3A6;
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.03em;
}

/* Section labels */
.section-label {
    font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em;
    color: #7A8B9A; font-weight: 600; margin: 22px 0 10px 0;
}

/* Agent card */
.agent-card {
    border-radius: 14px; padding: 16px 18px; background: #131A22;
    border: 1px solid #21303C; border-left: 4px solid var(--accent, #7A8B9A);
    height: 100%;
}
.agent-card .name { font-weight: 600; font-size: 0.95rem; color: #E6EDF3; }
.agent-card .badge {
    float: right; font-size: 0.68rem; padding: 2px 8px; border-radius: 999px;
    font-weight: 600; background: var(--badge-bg, #333); color: var(--badge-fg, #fff);
}
.agent-card .label {
    font-family: 'Space Grotesk', sans-serif; font-size: 1.35rem; font-weight: 700;
    color: var(--accent, #E6EDF3); margin-top: 10px;
}
.agent-card .dim { font-size: 0.72rem; color: #7A8B9A; text-transform: uppercase; letter-spacing: 0.05em; }
.agent-card .conf { font-size: 0.8rem; color: #9FB0C0; margin-top: 4px; }
.agent-card .reasoning { font-size: 0.82rem; color: #B8C4CE; margin-top: 10px; line-height: 1.4; }
.agent-card .meta { font-size: 0.7rem; color: #5D6D7A; margin-top: 10px; border-top: 1px solid #21303C; padding-top: 8px; }

/* Recommendation card */
.rec-card {
    border-radius: 16px; padding: 24px 28px; margin-top: 8px;
    background: linear-gradient(135deg, var(--rec-bg, #16261F) 0%, #131A22 100%);
    border: 1px solid var(--rec-border, #2A4A3D);
}
.rec-card .action {
    font-family: 'Space Grotesk', sans-serif; font-size: 1.6rem; font-weight: 700;
    color: var(--rec-accent, #22D3A6);
}
.rec-card .reasoning { color: #C7D3DC; margin-top: 8px; font-size: 0.92rem; line-height: 1.5; }
.rec-card .cites { margin-top: 12px; font-size: 0.78rem; color: #7A8B9A; }
.rec-card .cites code { background: #0B0F14; padding: 2px 6px; border-radius: 6px; color: #22D3A6; }

/* Profile comparison chip */
.profile-chip {
    border-radius: 12px; padding: 14px; background: #131A22; border: 1px solid #21303C; text-align: center;
}
.profile-chip .pname { font-weight: 600; text-transform: capitalize; color: #E6EDF3; font-size: 0.85rem; }
.profile-chip .paction { font-size: 0.85rem; color: #22D3A6; margin-top: 6px; font-weight: 600; }
.profile-chip .pconf { font-size: 0.72rem; color: #7A8B9A; margin-top: 4px; }

.stButton>button {
    border-radius: 10px; font-weight: 600; border: none;
    background: linear-gradient(90deg, #22D3A6, #14B88A); color: #0B0F14;
    padding: 0.6rem 1.4rem;
}
</style>
""", unsafe_allow_html=True)


def badge_style(o):
    """Returns (accent_color, badge_bg, badge_fg, badge_text) for an agent output."""
    if o.degraded:
        return "#F59E0B", "rgba(245,158,11,0.15)", "#F59E0B", "DEGRADED"
    bullish = {"BULLISH", "POSITIVE", "VOLUME_SPIKE", "GROUNDED"}
    bearish = {"BEARISH", "NEGATIVE"}
    if o.label in bullish:
        return "#22D3A6", "rgba(34,211,166,0.15)", "#22D3A6", "OK"
    if o.label in bearish:
        return "#F87171", "rgba(248,113,113,0.15)", "#F87171", "OK"
    return "#7DD3FC", "rgba(125,211,252,0.15)", "#7DD3FC", "OK"


def render_agent_card(o):
    accent, badge_bg, badge_fg, badge_text = badge_style(o)
    cites = ""
    if o.citations:
        cites = f'<div class="meta">📎 {", ".join(o.citations)}</div>'
    html = f"""
    <div class="agent-card" style="--accent:{accent}; --badge-bg:{badge_bg}; --badge-fg:{badge_fg};">
        <span class="badge">{badge_text}</span>
        <div class="name">{o.agent}</div>
        <div class="dim">{o.dimension.replace('_', ' ')}</div>
        <div class="label">{o.label}</div>
        <div class="conf">confidence {o.confidence:.2f}</div>
        <div class="reasoning">{o.reasoning}</div>
        <div class="meta">⚡ {o.latency_ms:.1f} ms</div>
        {cites}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_recommendation(synthesis):
    if synthesis["conflict"]:
        rec_bg, rec_border, rec_accent = "#2A1F14", "#4A3A2A", "#F59E0B"
    elif synthesis["action"].startswith("CONSIDER BUY"):
        rec_bg, rec_border, rec_accent = "#16261F", "#2A4A3D", "#22D3A6"
    elif synthesis["action"].startswith("CONSIDER REDUCE"):
        rec_bg, rec_border, rec_accent = "#2A1717", "#4A2A2A", "#F87171"
    else:
        rec_bg, rec_border, rec_accent = "#161C26", "#2A3A4A", "#7DD3FC"

    conflict_tag = ' ⚠️ CONFLICTING SIGNALS' if synthesis["conflict"] else ""
    cites_html = ""
    if synthesis["citations"]:
        chips = " ".join(f"<code>{c}</code>" for c in synthesis["citations"])
        cites_html = f'<div class="cites">📎 Cited sources: {chips}</div>'
    degraded_html = ""
    if synthesis["degraded_agents"]:
        degraded_html = (f'<div class="cites">⚠️ Degraded inputs: '
                          f'{", ".join(synthesis["degraded_agents"])} — confidence reduced, not fabricated.</div>')

    html = f"""
    <div class="rec-card" style="--rec-bg:{rec_bg}; --rec-border:{rec_border}; --rec-accent:{rec_accent};">
        <div class="action">{synthesis['action']}{conflict_tag}</div>
        <div class="reasoning">{synthesis['reasoning']}</div>
        {cites_html}
        {degraded_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
    st.progress(synthesis["confidence"], text=f"Confidence: {synthesis['confidence']:.0%}")


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>📊 Sentinel — Multi-Agent Investment Intelligence</h1>
    <p>Real-time signals, RAG-grounded reasoning, and risk-personalized recommendations — explainable at every step.</p>
    <span class="tag">PS-01 · HackVerse Sprint 1</span>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def get_rag_index():
    return RAGIndex(generate_filing_corpus())


rag_index = get_rag_index()
portfolio = generate_portfolio()

# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------
st.markdown('<div class="section-label">Configuration</div>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
with col1:
    ticker = st.selectbox("Ticker", TICKERS)
with col2:
    profile_name = st.selectbox("Risk profile", list(RISK_PROFILES.keys()), index=1)
with col3:
    scenario = st.selectbox("Market scenario", ["normal", "crash", "positive_news", "negative_news"])
with col4:
    degrade = st.selectbox("Simulate degraded feed", ["none", "momentum", "volume", "sentiment"])

run = st.button("▶  Run multi-agent analysis", type="primary")

st.markdown('<div class="section-label">Portfolio / Watchlist</div>', unsafe_allow_html=True)
st.dataframe(pd.DataFrame(list(portfolio.items()), columns=["Ticker", "Value (₹)"]),
             use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------
if run:
    crash = scenario == "crash"
    market_data = generate_market_data(ticker, crash=crash)
    news_sentiment = (
        "positive" if scenario == "positive_news" else "negative" if scenario == "negative_news" else "mixed"
    )
    news = generate_news(ticker, news_sentiment)

    outputs, synthesis = run_pipeline(
        ticker, market_data, news, rag_index, profile_name,
        simulate_degraded=None if degrade == "none" else degrade,
    )

    st.markdown('<div class="section-label">Agent Signals · Parallel Execution</div>', unsafe_allow_html=True)
    cols = st.columns(len(outputs))
    for c, o in zip(cols, outputs):
        with c:
            render_agent_card(o)

    st.markdown('<div class="section-label">Synthesized Recommendation</div>', unsafe_allow_html=True)
    render_recommendation(synthesis)

    st.markdown('<div class="section-label">Personalization Proof · Same Data, Every Profile</div>',
                unsafe_allow_html=True)
    comp_cols = st.columns(len(RISK_PROFILES))
    for c, pname in zip(comp_cols, RISK_PROFILES.keys()):
        _, syn2 = run_pipeline(ticker, market_data, news, rag_index, pname,
                                simulate_degraded=None if degrade == "none" else degrade)
        with c:
            st.markdown(f"""
            <div class="profile-chip">
                <div class="pname">{pname}{' ★' if pname == profile_name else ''}</div>
                <div class="paction">{syn2['action']}</div>
                <div class="pconf">confidence {syn2['confidence']:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

    forward_return_mock = random.uniform(-8, 8)
    row = log_session(ticker, profile_name, outputs, synthesis, forward_return_mock, portfolio)
    st.markdown('<div class="section-label">Session Metrics</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Avg agent latency", f"{row['avg_agent_latency_ms']} ms")
    m2.metric("Portfolio concentration (HHI)", row["portfolio_concentration_hhi"])
    m3.metric("Mock 30d forward return", f"{row['mock_30d_forward_return_pct']}%")
    m4.metric("Directional accuracy proxy", "✅" if row["directional_accuracy_proxy"] else "—")

st.markdown('<div class="section-label">Historical Performance Log</div>', unsafe_allow_html=True)
if os.path.exists(LOG_PATH):
    st.dataframe(pd.read_csv(LOG_PATH), use_container_width=True, hide_index=True)
else:
    st.info("Run an analysis above to start logging.")
