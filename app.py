import os
import random

import pandas as pd
import streamlit as st

from pipeline import (
    RAGIndex,
    RISK_PROFILES,
    LOG_PATH,
    run_pipeline,
    log_session,
)
from synthetic_data import (
    TICKERS,
    generate_market_data,
    generate_news,
    generate_filing_corpus,
    generate_portfolio,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Spider-Sense | Financial Intelligence",
    page_icon="🕷️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background:
            radial-gradient(circle at 15% 5%, rgba(120, 30, 60, 0.16), transparent 28%),
            radial-gradient(circle at 85% 10%, rgba(30, 90, 120, 0.10), transparent 25%),
            #07090d;
        color: #f1f3f5;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3 {
        letter-spacing: -0.02em;
    }

    /* ---------- HEADER ---------- */

    .hero {
        padding: 28px 32px;
        border-radius: 22px;
        border: 1px solid rgba(255,255,255,0.08);
        background:
            linear-gradient(
                135deg,
                rgba(25, 27, 34, 0.96),
                rgba(12, 14, 19, 0.96)
            );
        box-shadow: 0 20px 60px rgba(0,0,0,0.35);
        margin-bottom: 24px;
    }

    .hero-title {
        font-size: 38px;
        font-weight: 800;
        margin: 0;
        color: #ffffff;
    }

    .hero-subtitle {
        color: #9da5b4;
        font-size: 15px;
        margin-top: 8px;
    }

    .hero-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        background: rgba(190, 35, 70, 0.14);
        border: 1px solid rgba(220, 55, 90, 0.28);
        color: #ff718f;
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 12px;
    }

    /* ---------- SECTION ---------- */

    .section-title {
        font-size: 21px;
        font-weight: 750;
        margin-top: 28px;
        margin-bottom: 14px;
        color: #f4f6f8;
    }

    .section-caption {
        color: #89919f;
        font-size: 13px;
        margin-top: -8px;
        margin-bottom: 16px;
    }

    /* ---------- KPI CARDS ---------- */

    .kpi {
        background: rgba(18, 21, 28, 0.92);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 17px;
        padding: 19px 20px;
        min-height: 105px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.18);
    }

    .kpi-label {
        color: #858e9d;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
    }

    .kpi-value {
        color: #ffffff;
        font-size: 26px;
        font-weight: 800;
        margin-top: 7px;
    }

    .kpi-small {
        color: #8c95a4;
        font-size: 12px;
        margin-top: 4px;
    }

    /* ---------- AGENT CARDS ---------- */

    .agent-card {
        background: rgba(17,20,27,0.95);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 18px;
        padding: 19px;
        min-height: 220px;
        box-shadow: 0 12px 30px rgba(0,0,0,0.20);
    }

    .agent-name {
        font-weight: 750;
        font-size: 15px;
        color: #f5f7fa;
    }

    .agent-status {
        display: inline-block;
        float: right;
        font-size: 10px;
        font-weight: 800;
        padding: 4px 8px;
        border-radius: 999px;
    }

    .status-ok {
        color: #6ee7b7;
        background: rgba(16,185,129,0.10);
        border: 1px solid rgba(16,185,129,0.20);
    }

    .status-bad {
        color: #fb7185;
        background: rgba(244,63,94,0.10);
        border: 1px solid rgba(244,63,94,0.20);
    }

    .agent-dimension {
        color: #9ba3b1;
        font-size: 12px;
        margin-top: 20px;
    }

    .agent-signal {
        font-size: 22px;
        font-weight: 800;
        margin-top: 3px;
        color: #ffffff;
    }

    .agent-reason {
        color: #9da5b3;
        font-size: 12px;
        line-height: 1.55;
        margin-top: 13px;
    }

    .agent-meta {
        color: #6f7887;
        font-size: 10px;
        margin-top: 14px;
    }

    /* ---------- RECOMMENDATION ---------- */

    .recommendation {
        background:
            linear-gradient(
                135deg,
                rgba(28, 32, 42, 0.98),
                rgba(13, 16, 22, 0.98)
            );
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 22px;
        padding: 28px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.30);
    }

    .recommendation-label {
        color: #8d96a5;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 11px;
        font-weight: 800;
    }

    .recommendation-action {
        font-size: 34px;
        font-weight: 850;
        margin-top: 6px;
        color: #ffffff;
    }

    .recommendation-text {
        color: #b2bac7;
        font-size: 14px;
        line-height: 1.7;
        margin-top: 13px;
    }

    /* ---------- PROFILE ---------- */

    .profile-card {
        background: rgba(17,20,27,0.94);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 17px;
        padding: 18px;
        min-height: 125px;
    }

    .profile-name {
        color: #8e97a6;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
    }

    .profile-action {
        color: white;
        font-size: 19px;
        font-weight: 800;
        margin-top: 9px;
    }

    .profile-confidence {
        color: #78818f;
        font-size: 11px;
        margin-top: 8px;
    }

    /* ---------- NEWS ---------- */

    .news-card {
        background: rgba(17,20,27,0.94);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px;
        padding: 15px 17px;
        margin-bottom: 10px;
        color: #c6ccd5;
        font-size: 13px;
        line-height: 1.5;
    }

    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;
        color: #555d69;
        font-size: 11px;
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid rgba(255,255,255,0.05);
    }

    /* ---------- STREAMLIT CLEANUP ---------- */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    [data-testid="stDecoration"] {
        display: none;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">🕷️ SPIDER-SENSE · AI FINANCIAL INTELLIGENCE</div>
        <div class="hero-title">Autonomous Investment Intelligence</div>
        <div class="hero-subtitle">
            Multi-agent market analysis · Explainable signals · Personalized risk intelligence
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# RAG + PORTFOLIO
# ============================================================

@st.cache_resource
def get_rag_index():
    return RAGIndex(generate_filing_corpus())


rag_index = get_rag_index()
portfolio = generate_portfolio()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🕷️ Spider-Sense")

    st.caption("Investment Intelligence Console")

    st.divider()

    st.markdown("### Analysis Configuration")

    ticker = st.selectbox(
        "Target ticker",
        TICKERS,
    )

    profile_name = st.selectbox(
        "Investor profile",
        list(RISK_PROFILES.keys()),
        index=1,
    )

    scenario = st.selectbox(
        "Market scenario",
        [
            "normal",
            "crash",
            "positive_news",
            "negative_news",
        ],
    )

    degrade = st.selectbox(
        "Data degradation",
        [
            "none",
            "momentum",
            "volume",
            "sentiment",
        ],
    )

    st.divider()

    st.markdown("### System Status")

    st.success("● Market feed — ONLINE")
    st.success("● RAG knowledge base — ONLINE")
    st.success("● Agent orchestration — READY")

    if degrade != "none":
        st.warning(f"⚠ Simulating {degrade} degradation")

    st.divider()

    run = st.button(
        "▶  RUN INTELLIGENCE ANALYSIS",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# TOP KPI ROW
# ============================================================

total_value = sum(portfolio.values())

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">Portfolio Value</div>
            <div class="kpi-value">₹{total_value:,.0f}</div>
            <div class="kpi-small">Synthetic demonstration portfolio</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k2:
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">Tracked Assets</div>
            <div class="kpi-value">{len(TICKERS)}</div>
            <div class="kpi-small">Market intelligence universe</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k3:
    st.markdown(
        """
        <div class="kpi">
            <div class="kpi-label">Intelligence Agents</div>
            <div class="kpi-value">4</div>
            <div class="kpi-small">Parallel analytical signals</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k4:
    status_text = "DEGRADED" if degrade != "none" else "OPERATIONAL"

    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">System State</div>
            <div class="kpi-value">{status_text}</div>
            <div class="kpi-small">Graceful degradation enabled</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# PORTFOLIO
# ============================================================

st.markdown(
    '<div class="section-title">📊 Portfolio Intelligence</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-caption">Current allocation across the monitored universe</div>',
    unsafe_allow_html=True,
)

portfolio_df = pd.DataFrame(
    list(portfolio.items()),
    columns=["Ticker", "Value (₹)"],
)

portfolio_df["Allocation"] = (
    portfolio_df["Value (₹)"] / portfolio_df["Value (₹)"].sum() * 100
).round(1)

portfolio_df["Value (₹)"] = portfolio_df["Value (₹)"].map(
    lambda x: f"₹{x:,.0f}"
)

p1, p2 = st.columns([1.4, 1])

with p1:
    st.dataframe(
        portfolio_df,
        use_container_width=True,
        hide_index=True,
    )

with p2:
    chart_df = generate_market_data(ticker)

    prices = pd.DataFrame(
        {
            "Day": range(1, len(chart_df["prices"]) + 1),
            "Price": chart_df["prices"],
        }
    )

    st.markdown(f"**{ticker} · 30-Day Synthetic Price Path**")
    st.line_chart(
        prices.set_index("Day"),
        use_container_width=True,
    )


# ============================================================
# ANALYSIS
# ============================================================

if run:

    with st.spinner("🧠 Agents are analyzing market intelligence..."):

        crash = scenario == "crash"

        market_data = generate_market_data(
            ticker,
            crash=crash,
        )

        news_sentiment = (
            "positive"
            if scenario == "positive_news"
            else "negative"
            if scenario == "negative_news"
            else "mixed"
        )

        news = generate_news(
            ticker,
            news_sentiment,
        )

        outputs, synthesis = run_pipeline(
            ticker,
            market_data,
            news,
            rag_index,
            profile_name,
            simulate_degraded=None if degrade == "none" else degrade,
        )


    # ========================================================
    # ANALYSIS HEADER
    # ========================================================

    st.markdown(
        f"""
        <div class="section-title">
            🧠 Intelligence Analysis · {ticker}
        </div>
        <div class="section-caption">
            {profile_name} profile · {scenario.replace("_", " ").title()} scenario
        </div>
        """,
        unsafe_allow_html=True,
    )


    # ========================================================
    # AGENT SIGNALS
    # ========================================================

    agent_cols = st.columns(len(outputs))

    for c, o in zip(agent_cols, outputs):

        status_class = "status-bad" if o.degraded else "status-ok"
        status_text = "DEGRADED" if o.degraded else "ONLINE"

        with c:

            citations = ""

            if o.citations:
                citations = (
                    "<div class='agent-meta'>"
                    "Sources: "
                    + ", ".join(o.citations)
                    + "</div>"
                )

            st.markdown(
                f"""
                <div class="agent-card">

                    <div class="agent-name">
                        {o.agent}
                        <span class="agent-status {status_class}">
                            {status_text}
                        </span>
                    </div>

                    <div class="agent-dimension">
                        {o.dimension}
                    </div>

                    <div class="agent-signal">
                        {o.label}
                    </div>

                    <div class="agent-reason">
                        {o.reasoning}
                    </div>

                    <div class="agent-meta">
                        Confidence: {o.confidence}
                        · Latency: {o.latency_ms:.1f} ms
                    </div>

                    {citations}

                </div>
                """,
                unsafe_allow_html=True,
            )


    # ========================================================
    # RECOMMENDATION
    # ========================================================

    st.markdown(
        '<div class="section-title">🎯 Synthesis Engine</div>',
        unsafe_allow_html=True,
    )

    conflict_text = ""

    if synthesis["conflict"]:
        conflict_text = " · ⚠ CONFLICTING SIGNALS"

    st.markdown(
        f"""
        <div class="recommendation">

            <div class="recommendation-label">
                Personalized recommendation{conflict_text}
            </div>

            <div class="recommendation-action">
                {synthesis["action"]}
            </div>

            <div class="recommendation-text">
                {synthesis["reasoning"]}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(
        synthesis["confidence"],
        text=f"Model confidence · {synthesis['confidence']:.0%}",
    )

    if synthesis["citations"]:
        st.caption(
            "📎 Evidence: " + ", ".join(synthesis["citations"])
        )

    if synthesis["degraded_agents"]:

        st.warning(
            f"Degraded inputs: {', '.join(synthesis['degraded_agents'])}. "
            "Recommendation issued with reduced confidence rather than fabricated data."
        )


    # ========================================================
    # NEWS + MARKET
    # ========================================================

    st.markdown(
        '<div class="section-title">📰 Market Intelligence Feed</div>',
        unsafe_allow_html=True,
    )

    n1, n2 = st.columns([1, 1])

    with n1:

        st.markdown("**Synthetic market signals**")

        price_change = (
            (market_data["prices"][-1] / market_data["prices"][0]) - 1
        ) * 100

        m1, m2 = st.columns(2)

        with m1:
            st.metric(
                "30D Price",
                f"₹{market_data['prices'][-1]:,.0f}",
                f"{price_change:+.2f}%",
            )

        with m2:
            st.metric(
                "Latest Volume",
                f"{market_data['volumes'][-1]:,.0f}",
            )

        st.line_chart(
            pd.DataFrame(
                {"Price": market_data["prices"]}
            ),
            use_container_width=True,
        )

    with n2:

        st.markdown("**News sentiment signals**")

        if news:

            for item in news:

                st.markdown(
                    f"""
                    <div class="news-card">
                        ◉ {item}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        else:

            st.info("No news signals returned for this scenario.")


    # ========================================================
    # PERSONALIZATION
    # ========================================================

    st.markdown(
        '<div class="section-title">👤 Personalization Engine</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-caption">'
        "Same market evidence · Different risk preferences · Different decisions"
        "</div>",
        unsafe_allow_html=True,
    )

    comp_cols = st.columns(len(RISK_PROFILES))

    for c, pname in zip(comp_cols, RISK_PROFILES.keys()):

        _, syn2 = run_pipeline(
            ticker,
            market_data,
            news,
            rag_index,
            pname,
            simulate_degraded=None if degrade == "none" else degrade,
        )

        with c:

            st.markdown(
                f"""
                <div class="profile-card">

                    <div class="profile-name">
                        {pname}
                    </div>

                    <div class="profile-action">
                        {syn2["action"]}
                    </div>

                    <div class="profile-confidence">
                        Confidence · {syn2["confidence"]:.0%}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


    # ========================================================
    # PERFORMANCE LOG
    # ========================================================

    forward_return_mock = random.uniform(-8, 8)

    row = log_session(
        ticker,
        profile_name,
        outputs,
        synthesis,
        forward_return_mock,
        portfolio,
    )

    with st.expander("📈 Session performance record"):

        st.json(row)


# ============================================================
# HISTORICAL LOG
# ============================================================

st.markdown(
    '<div class="section-title">🗂 Intelligence History</div>',
    unsafe_allow_html=True,
)

if os.path.exists(LOG_PATH):

    history = pd.read_csv(LOG_PATH)

    st.dataframe(
        history,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info(
        "No historical analysis sessions yet. "
        "Run an analysis to populate the intelligence log."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        SPIDER-SENSE · HackVerse Sprint 1 · Explainable Multi-Agent Financial Intelligence
        <br>
        Synthetic market data used for demonstration purposes.
    </div>
    """,
    unsafe_allow_html=True,
)
