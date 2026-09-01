import os
import random

import pandas as pd
import plotly.graph_objects as go
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
    page_title="Spider-Sense | AI Financial Intelligence",
    page_icon="🕷️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PREMIUM UI
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       GLOBAL
       ====================================================== */

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(
                circle at 10% 0%,
                rgba(170, 25, 65, 0.16),
                transparent 25%
            ),
            radial-gradient(
                circle at 90% 5%,
                rgba(40, 100, 160, 0.10),
                transparent 25%
            ),
            #06080c;
        color: #f4f5f7;
    }

    .block-container {
        max-width: 1550px;
        padding-top: 1.4rem;
        padding-bottom: 4rem;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    [data-testid="stDecoration"] {
        display: none;
    }

    /* ======================================================
       SIDEBAR
       ====================================================== */

    section[data-testid="stSidebar"] {
        background: #090b10;
        border-right: 1px solid rgba(255,255,255,0.07);
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    /* ======================================================
       HERO
       ====================================================== */

    .hero {
        position: relative;
        overflow: hidden;
        border-radius: 26px;
        padding: 34px 36px;
        margin-bottom: 22px;

        background:
            linear-gradient(
                135deg,
                rgba(24,27,36,0.98),
                rgba(9,11,16,0.98)
            );

        border: 1px solid rgba(255,255,255,0.09);

        box-shadow:
            0 25px 80px rgba(0,0,0,0.38);
    }

    .hero:after {
        content: "";
        position: absolute;
        width: 260px;
        height: 260px;
        right: -100px;
        top: -120px;
        border-radius: 50%;

        background: rgba(190, 35, 70, 0.12);

        filter: blur(55px);
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;

        padding: 7px 12px;
        border-radius: 999px;

        background: rgba(210,45,85,0.11);
        border: 1px solid rgba(230,70,105,0.25);

        color: #ff7894;

        font-size: 11px;
        font-weight: 800;
        letter-spacing: 0.10em;

        margin-bottom: 13px;
    }

    .hero-title {
        font-size: 39px;
        line-height: 1.1;
        font-weight: 800;
        letter-spacing: -0.04em;

        color: #ffffff;
    }

    .hero-title span {
        color: #ff587a;
    }

    .hero-subtitle {
        margin-top: 10px;

        color: #9199a8;
        font-size: 14px;
        line-height: 1.6;
    }

    /* ======================================================
       SECTION HEADERS
       ====================================================== */

    .section {
        margin-top: 30px;
        margin-bottom: 14px;
    }

    .section-title {
        color: #f5f6f8;
        font-size: 21px;
        font-weight: 800;
        letter-spacing: -0.025em;
    }

    .section-subtitle {
        color: #727b89;
        font-size: 12px;
        margin-top: 4px;
    }

    /* ======================================================
       KPI
       ====================================================== */

    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
    }

    .kpi {
        min-height: 112px;

        padding: 18px 19px;

        border-radius: 18px;

        background:
            linear-gradient(
                145deg,
                rgba(22,25,33,0.96),
                rgba(13,15,21,0.96)
            );

        border: 1px solid rgba(255,255,255,0.07);

        box-shadow:
            0 12px 35px rgba(0,0,0,0.18);
    }

    .kpi-label {
        color: #727b89;
        font-size: 10px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.12em;
    }

    .kpi-value {
        margin-top: 8px;

        color: #ffffff;
        font-size: 27px;
        font-weight: 800;
        letter-spacing: -0.03em;
    }

    .kpi-detail {
        margin-top: 4px;

        color: #747d8b;
        font-size: 10px;
    }

    /* ======================================================
       CONTROL BAR
       ====================================================== */

    .control-card {
        background: rgba(15,18,24,0.94);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 18px;
        padding: 15px 17px;
    }

    /* ======================================================
       AGENT CARDS
       ====================================================== */

    .agent-card {
        min-height: 235px;

        padding: 20px;

        border-radius: 19px;

        background:
            linear-gradient(
                145deg,
                rgba(20,23,31,0.97),
                rgba(11,13,18,0.97)
            );

        border: 1px solid rgba(255,255,255,0.075);

        box-shadow:
            0 14px 38px rgba(0,0,0,0.20);

        transition: transform 0.2s ease;
    }

    .agent-card:hover {
        transform: translateY(-2px);
    }

    .agent-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .agent-name {
        color: #f7f8fa;
        font-size: 14px;
        font-weight: 800;
    }

    .online {
        color: #67e8a5;
        background: rgba(16,185,129,0.10);
        border: 1px solid rgba(16,185,129,0.20);
    }

    .degraded {
        color: #fb7185;
        background: rgba(244,63,94,0.10);
        border: 1px solid rgba(244,63,94,0.20);
    }

    .status {
        padding: 4px 8px;
        border-radius: 999px;

        font-size: 9px;
        font-weight: 800;
        letter-spacing: 0.08em;
    }

    .agent-dimension {
        color: #717a88;
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 0.10em;

        margin-top: 23px;
    }

    .agent-signal {
        color: #ffffff;

        font-size: 23px;
        font-weight: 800;

        margin-top: 5px;
    }

    .agent-reason {
        color: #929aa8;

        font-size: 11px;
        line-height: 1.6;

        margin-top: 12px;

        min-height: 54px;
    }

    .agent-meta {
        color: #646d7b;

        font-size: 9px;

        margin-top: 15px;
    }

    /* ======================================================
       SYNTHESIS
       ====================================================== */

    .synthesis {
        position: relative;
        overflow: hidden;

        padding: 30px;

        border-radius: 23px;

        background:
            radial-gradient(
                circle at 90% 10%,
                rgba(210,40,75,0.12),
                transparent 28%
            ),
            linear-gradient(
                135deg,
                rgba(27,30,40,0.98),
                rgba(11,13,18,0.98)
            );

        border: 1px solid rgba(255,255,255,0.09);

        box-shadow:
            0 25px 65px rgba(0,0,0,0.30);
    }

    .synthesis-label {
        color: #7e8795;

        font-size: 10px;
        font-weight: 800;

        text-transform: uppercase;
        letter-spacing: 0.13em;
    }

    .synthesis-action {
        color: #ffffff;

        font-size: 36px;
        font-weight: 850;

        letter-spacing: -0.04em;

        margin-top: 6px;
    }

    .synthesis-reason {
        color: #aab2bf;

        font-size: 13px;
        line-height: 1.75;

        margin-top: 13px;
    }

    /* ======================================================
       NEWS
       ====================================================== */

    .news {
        padding: 14px 16px;

        margin-bottom: 9px;

        border-radius: 14px;

        background: rgba(20,23,30,0.90);

        border: 1px solid rgba(255,255,255,0.065);

        color: #b8c0cc;

        font-size: 11px;

        line-height: 1.55;
    }

    /* ======================================================
       PROFILE
       ====================================================== */

    .profile {
        padding: 19px;

        min-height: 130px;

        border-radius: 17px;

        background: rgba(17,20,27,0.95);

        border: 1px solid rgba(255,255,255,0.07);
    }

    .profile-name {
        color: #747d8a;

        font-size: 9px;
        font-weight: 800;

        text-transform: uppercase;
        letter-spacing: 0.11em;
    }

    .profile-action {
        color: #ffffff;

        font-size: 19px;
        font-weight: 800;

        margin-top: 9px;
    }

    .profile-confidence {
        color: #687180;

        font-size: 10px;

        margin-top: 8px;
    }

    /* ======================================================
       EVIDENCE
       ====================================================== */

    .evidence {
        padding: 16px;

        border-radius: 15px;

        background: rgba(16,19,25,0.90);

        border: 1px solid rgba(255,255,255,0.06);

        color: #aeb6c2;

        font-size: 11px;
        line-height: 1.6;
    }

    /* ======================================================
       FOOTER
       ====================================================== */

    .footer {
        margin-top: 50px;
        padding-top: 20px;

        border-top: 1px solid rgba(255,255,255,0.05);

        text-align: center;

        color: #4f5763;

        font-size: 9px;

        letter-spacing: 0.06em;
    }

    /* ======================================================
       BUTTONS
       ====================================================== */

    div.stButton > button {
        border-radius: 12px;

        font-weight: 800;

        min-height: 44px;

        transition: all 0.2s ease;
    }

    div.stButton > button:hover {
        transform: translateY(-1px);
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

        <div class="hero-badge">
            🕷 SPIDER-SENSE · AUTONOMOUS FINANCIAL INTELLIGENCE
        </div>

        <div class="hero-title">
            Intelligence before <span>investment.</span>
        </div>

        <div class="hero-subtitle">
            A multi-agent financial intelligence system that combines
            market signals, sentiment, risk analysis and document intelligence
            to produce explainable, personalized decisions.
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# BACKEND
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

    st.markdown(
        """
        <div style="
            font-size:25px;
            font-weight:800;
            color:white;
            margin-bottom:2px;
        ">
            🕷️ Spider-Sense
        </div>

        <div style="
            color:#707987;
            font-size:11px;
            margin-bottom:20px;
        ">
            AI Financial Intelligence Console
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("#### ANALYSIS CONFIGURATION")

    ticker = st.selectbox(
        "Target asset",
        TICKERS,
    )

    profile_name = st.selectbox(
        "Investor risk profile",
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
        "Data-feed simulation",
        [
            "none",
            "momentum",
            "volume",
            "sentiment",
        ],
    )

    st.divider()

    st.markdown("#### SYSTEM HEALTH")

    if degrade == "none":
        st.success("● All systems operational")
    else:
        st.warning(f"⚠ {degrade.upper()} feed degraded")

    st.caption("Market intelligence      ONLINE")
    st.caption("RAG knowledge base       ONLINE")
    st.caption("Agent orchestration      READY")

    st.divider()

    run = st.button(
        "▶  RUN INTELLIGENCE",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# KPI ROW
# ============================================================

total_value = sum(portfolio.values())

st.markdown(
    """
    <div class="section">
        <div class="section-title">Command Center</div>
        <div class="section-subtitle">
            Real-time state of the intelligence environment
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">Portfolio Value</div>
            <div class="kpi-value">₹{total_value:,.0f}</div>
            <div class="kpi-detail">Synthetic demonstration portfolio</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k2:
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">Assets Monitored</div>
            <div class="kpi-value">{len(TICKERS)}</div>
            <div class="kpi-detail">Active intelligence universe</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k3:
    st.markdown(
        """
        <div class="kpi">
            <div class="kpi-label">AI Agents</div>
            <div class="kpi-value">4</div>
            <div class="kpi-detail">Parallel analytical engines</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k4:

    system_state = (
        "DEGRADED"
        if degrade != "none"
        else "OPERATIONAL"
    )

    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">System State</div>
            <div class="kpi-value">{system_state}</div>
            <div class="kpi-detail">
                Graceful degradation enabled
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# PORTFOLIO + MARKET
# ============================================================

st.markdown(
    """
    <div class="section">
        <div class="section-title">Market Intelligence</div>
        <div class="section-subtitle">
            Portfolio exposure and synthetic market trajectory
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

portfolio_df = pd.DataFrame(
    list(portfolio.items()),
    columns=["Ticker", "Value"],
)

portfolio_df["Allocation"] = (
    portfolio_df["Value"]
    / portfolio_df["Value"].sum()
    * 100
)

market_preview = generate_market_data(ticker)

latest_price = market_preview["prices"][-1]

price_change = (
    latest_price
    / market_preview["prices"][0]
    - 1
) * 100


m1, m2 = st.columns([1, 1.65])

with m1:

    st.markdown(
        f"""
        <div class="kpi" style="margin-bottom:12px;">
            <div class="kpi-label">{ticker} · LAST PRICE</div>
            <div class="kpi-value">₹{latest_price:,.2f}</div>
            <div class="kpi-detail">
                30-day movement · {price_change:+.2f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.dataframe(
        portfolio_df.assign(
            Value=portfolio_df["Value"].map(
                lambda x: f"₹{x:,.0f}"
            ),
            Allocation=portfolio_df["Allocation"].map(
                lambda x: f"{x:.1f}%"
            ),
        ),
        use_container_width=True,
        hide_index=True,
    )


with m2:

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(market_preview["prices"]) + 1)),
            y=market_preview["prices"],
            mode="lines",
            line=dict(
                width=3,
            ),
            fill="tozeroy",
            fillcolor="rgba(255,70,110,0.08)",
            hovertemplate="Day %{x}<br>₹%{y:,.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"{ticker} · 30-Day Price Intelligence",
        height=340,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color="#9aa2af",
            size=11,
        ),
        xaxis=dict(
            title=None,
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            title=None,
            showgrid=True,
            gridcolor="rgba(255,255,255,0.05)",
            zeroline=False,
        ),
        hoverlabel=dict(
            bgcolor="#11151d",
            font_size=11,
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False},
    )


# ============================================================
# ANALYSIS
# ============================================================

if run:

    crash = scenario == "crash"

    news_sentiment = (
        "positive"
        if scenario == "positive_news"
        else "negative"
        if scenario == "negative_news"
        else "mixed"
    )

    with st.status(
        "🧠 Running multi-agent intelligence...",
        expanded=True,
    ) as status:

        st.write("Initializing market intelligence...")
        market_data = generate_market_data(
            ticker,
            crash=crash,
        )

        st.write("Processing news intelligence...")
        news = generate_news(
            ticker,
            news_sentiment,
        )

        st.write("Executing analytical agents...")
        outputs, synthesis = run_pipeline(
            ticker,
            market_data,
            news,
            rag_index,
            profile_name,
            simulate_degraded=None
            if degrade == "none"
            else degrade,
        )

        status.update(
            label="✓ Intelligence analysis complete",
            state="complete",
            expanded=False,
        )


    # ========================================================
    # AGENTS
    # ========================================================

    st.markdown(
        f"""
        <div class="section">
            <div class="section-title">
                Neural Agent Grid · {ticker}
            </div>
            <div class="section-subtitle">
                Independent analytical agents executed in parallel
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    agent_cols = st.columns(len(outputs))

    for col, output in zip(agent_cols, outputs):

        status_class = (
            "degraded"
            if output.degraded
            else "online"
        )

        status_text = (
            "DEGRADED"
            if output.degraded
            else "ONLINE"
        )

        source_html = ""

        if output.citations:
            source_html = (
                "<div class='agent-meta'>"
                "Evidence · "
                + ", ".join(output.citations)
                + "</div>"
            )

        with col:

            st.markdown(
                f"""
                <div class="agent-card">

                    <div class="agent-header">

                        <div class="agent-name">
                            {output.agent}
                        </div>

                        <div class="status {status_class}">
                            {status_text}
                        </div>

                    </div>

                    <div class="agent-dimension">
                        {output.dimension}
                    </div>

                    <div class="agent-signal">
                        {output.label}
                    </div>

                    <div class="agent-reason">
                        {output.reasoning}
                    </div>

                    <div class="agent-meta">
                        Confidence · {output.confidence}
                        &nbsp;&nbsp;•&nbsp;&nbsp;
                        Latency · {output.latency_ms:.1f} ms
                    </div>

                    {source_html}

                </div>
                """,
                unsafe_allow_html=True,
            )


    # ========================================================
    # SYNTHESIS
    # ========================================================

    st.markdown(
        """
        <div class="section">
            <div class="section-title">Synthesis Engine</div>
            <div class="section-subtitle">
                Cross-agent reasoning and personalized decision
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    conflict = ""

    if synthesis["conflict"]:
        conflict = " · ⚠ CONFLICTING SIGNALS"

    st.markdown(
        f"""
        <div class="synthesis">

            <div class="synthesis-label">
                FINAL PERSONALIZED DECISION{conflict}
            </div>

            <div class="synthesis-action">
                {synthesis["action"]}
            </div>

            <div class="synthesis-reason">
                {synthesis["reasoning"]}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(
        synthesis["confidence"],
        text=f"Decision confidence · {synthesis['confidence']:.0%}",
    )

    if synthesis["degraded_agents"]:

        st.warning(
            "Graceful degradation active: "
            + ", ".join(synthesis["degraded_agents"])
            + ". Recommendation confidence has been reduced rather than fabricated."
        )


    # ========================================================
    # MARKET + NEWS
    # ========================================================

    st.markdown(
        """
        <div class="section">
            <div class="section-title">Evidence Layer</div>
            <div class="section-subtitle">
                Market movement, sentiment and supporting intelligence
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    e1, e2 = st.columns([1.15, 0.85])

    with e1:

        volume_df = pd.DataFrame(
            {
                "Day": range(
                    1,
                    len(market_data["volumes"]) + 1,
                ),
                "Volume": market_data["volumes"],
            }
        )

        fig2 = go.Figure()

        fig2.add_trace(
            go.Bar(
                x=volume_df["Day"],
                y=volume_df["Volume"],
                opacity=0.7,
                hovertemplate="Day %{x}<br>Volume %{y:,.0f}<extra></extra>",
            )
        )

        fig2.update_layout(
            title="Trading Volume Intelligence",
            height=280,
            margin=dict(l=10, r=10, t=50, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(
                color="#9aa2af",
                size=11,
            ),
            xaxis=dict(
                showgrid=False,
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="rgba(255,255,255,0.05)",
            ),
        )

        st.plotly_chart(
            fig2,
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with e2:

        st.markdown("**News intelligence**")

        if news:

            for item in news:

                st.markdown(
                    f"""
                    <div class="news">
                        ◉ &nbsp; {item}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        else:

            st.info(
                "No news signals returned for this scenario."
            )


    # ========================================================
    # PERSONALIZATION
    # ========================================================

    st.markdown(
        """
        <div class="section">
            <div class="section-title">Personalization Matrix</div>
            <div class="section-subtitle">
                Same evidence · different risk tolerance · different decisions
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    profile_cols = st.columns(
        len(RISK_PROFILES)
    )

    for col, pname in zip(
        profile_cols,
        RISK_PROFILES.keys(),
    ):

        _, profile_synthesis = run_pipeline(
            ticker,
            market_data,
            news,
            rag_index,
            pname,
            simulate_degraded=None
            if degrade == "none"
            else degrade,
        )

        with col:

            st.markdown(
                f"""
                <div class="profile">

                    <div class="profile-name">
                        {pname}
                    </div>

                    <div class="profile-action">
                        {profile_synthesis["action"]}
                    </div>

                    <div class="profile-confidence">
                        Confidence ·
                        {profile_synthesis["confidence"]:.0%}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


    # ========================================================
    # EXPLAINABILITY
    # ========================================================

    st.markdown(
        """
        <div class="section">
            <div class="section-title">Explainability & Evidence</div>
            <div class="section-subtitle">
                Why the system reached its decision
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    x1, x2 = st.columns(2)

    with x1:

        st.markdown(
            f"""
            <div class="evidence">
                <strong>Decision reasoning</strong><br><br>
                {synthesis["reasoning"]}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with x2:

        citations = synthesis.get(
            "citations",
            [],
        )

        if citations:

            citation_text = "<br>".join(
                f"• {c}"
                for c in citations
            )

        else:

            citation_text = (
                "No explicit citations returned."
            )

        st.markdown(
            f"""
            <div class="evidence">
                <strong>Supporting evidence</strong><br><br>
                {citation_text}
            </div>
            """,
            unsafe_allow_html=True,
        )


    # ========================================================
    # SESSION LOG
    # ========================================================

    forward_return_mock = random.uniform(
        -8,
        8,
    )

    row = log_session(
        ticker,
        profile_name,
        outputs,
        synthesis,
        forward_return_mock,
        portfolio,
    )

    with st.expander(
        "📈 View raw session record"
    ):
        st.json(row)


# ============================================================
# HISTORY
# ============================================================

st.markdown(
    """
    <div class="section">
        <div class="section-title">Intelligence History</div>
        <div class="section-subtitle">
            Previous analysis sessions
        </div>
    </div>
    """,
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
        "No historical sessions yet. "
        "Run an intelligence analysis to begin."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        SPIDER-SENSE · HACKVERSE · AUTONOMOUS FINANCIAL INTELLIGENCE
        <br><br>
        Synthetic market data is used for demonstration purposes.
    </div>
    """,
    unsafe_allow_html=True,
)
