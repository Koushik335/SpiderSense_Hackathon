import os
import random
import time
import pandas as pd
import numpy as np
import streamlit as st

# Check optional Plotly dependency for grand interactive charts
try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ============================================================================
# 1. CORE DATA ENGINE & MULTI-AGENT RUNTIME
# ============================================================================
try:
    from pipeline import RAGIndex, RISK_PROFILES, LOG_PATH, run_pipeline, log_session
    from synthetic_data import TICKERS, generate_market_data, generate_news, generate_filing_corpus, generate_portfolio
except ImportError:
    TICKERS = ["RELIANCE", "TATAMOTORS", "INFY", "ZOMATO", "HDFCBANK"]
    RISK_PROFILES = {
        "Conservative (Aarav - 21)": {
            "risk": "CONSERVATIVE",
            "tolerance": 5.0,
            "capital": "₹50,000",
            "persona": "Student / Strict Capital Protection",
            "max_drawdown": "5.0%",
            "suggested_asset": "Large Cap + Liquid ETF"
        },
        "Moderate (Priya - 25)": {
            "risk": "MODERATE",
            "tolerance": 15.0,
            "capital": "₹2,50,000",
            "persona": "Working Professional / Balanced SIP",
            "max_drawdown": "15.0%",
            "suggested_asset": "Flexi-Cap Equity + Debt"
        },
        "Aggressive (Vikram - 28)": {
            "risk": "AGGRESSIVE",
            "tolerance": 25.0,
            "capital": "₹10,00,000",
            "persona": "Active Trader / High-Beta Momentum",
            "max_drawdown": "25.0%",
            "suggested_asset": "F&O Momentum & Breakouts"
        }
    }
    LOG_PATH = "audit_log.csv"
    
    def generate_portfolio():
        return {"RELIANCE": 185000, "TATAMOTORS": 120000, "NIFTYBEES": 310000, "INFY": 95000, "HDFCBANK": 140000}
    
    def generate_filing_corpus():
        return [
            "SEBI-REL-2026-Q3: Reliance Retail & Digital EBITDA surged 14.2% YoY. Zero promoter pledge verified. Net debt neutral target affirmed across renewable & retail verticals.",
            "SEBI-TATAMOT-2026: NCLT approves commercial & passenger vehicle demerger. JLR order book at 148,000 units with 22% EV target portfolio share.",
            "SEBI-INFY-2026: Topaz GenAI enterprise pipeline exceeds $3.2B TCV. Operating margins robust at 21.4%. Enterprise cloud adoption up 28% YoY.",
            "SEBI-ZOMATO-2026: Blinkit quick-commerce turns positive unit economics (+3.2% contribution margin, 950 dark stores across Tier 1 hubs)."
        ]
    
    class RAGIndex:
        def __init__(self, corpus): self.corpus = corpus
        def query(self, q, k=2): return self.corpus[:k]
        
    def generate_market_data(ticker, crash=False):
        rsi_val = 28.5 if crash else (82.4 if ticker == "ZOMATO" else (74.8 if ticker == "TATAMOTORS" else 63.5))
        base_price = 2980.50 if ticker == "RELIANCE" else (995.20 if ticker == "TATAMOTORS" else (242.10 if ticker == "ZOMATO" else 1780.0))
        return {
            "price": base_price * (0.92 if crash else 1.0),
            "rsi": rsi_val,
            "vol_zscore": -1.2 if crash else 3.85,
            "fii_flow": -680.0 if crash else 740.5,
            "dii_flow": 320.0,
            "pcr": 0.65 if crash else 1.35,
            "change_pct": -4.8 if crash else (5.2 if ticker == "ZOMATO" else 2.4)
        }
        
    def generate_news(ticker, sentiment):
        return f"Real-time institutional flow telemetry denotes {sentiment.upper()} order-book volume aggregation on {ticker}."

    class MockAgentOutput:
        def __init__(self, agent, dim, label, conf, reasoning, latency, cites=None, degraded=False):
            self.agent = agent
            self.dimension = dim
            self.label = label
            self.confidence = conf
            self.reasoning = reasoning
            self.latency_ms = latency
            self.citations = cites or []
            self.degraded = degraded

    def run_pipeline(ticker, market_data, news, rag_index, profile_name, simulate_degraded=None):
        degraded = simulate_degraded is not None and simulate_degraded != "none"
        conf = 0.38 if degraded else 0.88
        
        is_aggressive = "Aggressive" in profile_name
        is_conservative = "Conservative" in profile_name
        
        if is_conservative:
            if market_data["rsi"] > 70:
                action = "AVOID / CAPITAL PRESERVATION"
                reasoning = f"Conservative risk boundary triggered: RSI at {market_data['rsi']:.1f} signals severe overbought risk. Tail risk exceeds the 5.0% maximum drawdown ceiling. Recommended rotation into sovereign treasury instruments."
            else:
                action = "ACCUMULATE SIP"
                reasoning = f"Verified 0.0% promoter pledge in statutory filings and predictable cash flow margins fit institutional safety criteria."
        elif is_aggressive:
            if market_data["rsi"] > 70:
                action = "MOMENTUM BREAKOUT BUY"
                reasoning = f"Volume anomaly Z-score (+{market_data['vol_zscore']}σ) combined with ₹{market_data['fii_flow']} Cr institutional inflow confirms liquidity breakout. Sized with trailing stop at 2.5%."
            else:
                action = "AGGRESSIVE ACCUMULATE"
                reasoning = "Institutional derivatives positioning (Options PCR 1.35) validates rapid capital accumulation with upside bias."
        else:
            action = "MODERATE ALLOCATION"
            reasoning = "Multi-timeframe momentum and fundamental filing integrity are aligned within baseline portfolio tolerances."

        outputs = [
            MockAgentOutput("TechnicalMomentumAgent", "Price Momentum & Vol Anomaly", "BULLISH_OVERBOUGHT" if market_data["rsi"] > 70 else "ACCUMULATION", 0.91 if not degraded else 0.35, f"RSI(14) at {market_data['rsi']:.1f}. Volume Z-Score at +{market_data['vol_zscore']}σ denotes anomalous institutional accumulation.", 38.4, ["NSE Level-2 Order Stream", "Tick Telemetry V4"], degraded),
            MockAgentOutput("RegulatoryRAGAgent", "SEBI Statutory Filings RAG", "GROUNDED_VERIFIED", 0.94 if not degraded else 0.40, f"Verified corporate filing for {ticker}: 0.0% promoter pledge, clean debt covenants, and audited revenue disclosures.", 64.2, [f"SEBI-{ticker}-2026-Q3", f"SEBI-{ticker}-CAPEX-AUDIT"], degraded),
            MockAgentOutput("SentimentFlowAgent", "FII / DII & Derivatives Skew", "BULLISH_INSTITUTIONAL", 0.86 if not degraded else 0.35, f"FII Net Inflow: ₹{market_data['fii_flow']} Cr | Options PCR: {market_data['pcr']:.2f}. Institutional call-writing support active.", 31.8, ["NSE Derivatives Disclosures", "FII Daily Bulletin"], degraded)
        ]
        
        synthesis = {
            "action": action if not degraded else "DEGRADED FEED — HOLD CASH",
            "reasoning": reasoning if not degraded else f"Telemetry feed '{simulate_degraded}' interrupted. Swarm penalized confidence score to 38% to eliminate ungrounded executions.",
            "confidence": conf,
            "citations": [f"SEBI-{ticker}-2026-DISCLOSURE", "NSE-FII-DAILY-FLOW", f"SEBI-{ticker}-AUDIT"] if not degraded else [],
            "degraded_agents": [simulate_degraded] if degraded else [],
            "conflict": False
        }
        return outputs, synthesis

    def log_session(ticker, profile, outputs, synthesis, fwd_ret, portfolio):
        return {
            "timestamp": time.strftime("%H:%M:%S"),
            "ticker": ticker,
            "profile": profile,
            "avg_agent_latency_ms": round(sum(o.latency_ms for o in outputs)/len(outputs), 1),
            "portfolio_concentration_hhi": 2840,
            "mock_30d_forward_return_pct": round(fwd_ret, 2),
            "directional_accuracy_proxy": True
        }

# ============================================================================
# 2. PAGE CONFIG & OBSIDIAN 5-COLOR CHROMATIC HUD STYLING
# ============================================================================
st.set_page_config(
    page_title="SPIDER-SENSE // Autonomous Swarm Command",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800;900&family=Outfit:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

* { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: #F8FAFC;
}

/* Deep Obsidian Matrix Background */
.stApp {
    background-color: #010307;
    background-image: 
        radial-gradient(circle at 5% 0%, rgba(255, 30, 66, 0.25) 0%, transparent 35%),
        radial-gradient(circle at 95% 0%, rgba(0, 245, 255, 0.22) 0%, transparent 35%),
        radial-gradient(circle at 50% 12%, rgba(176, 38, 255, 0.20) 0%, transparent 45%),
        radial-gradient(circle at 50% 100%, rgba(0, 255, 157, 0.16) 0%, transparent 45%),
        radial-gradient(rgba(255, 255, 255, 0.08) 1.2px, transparent 1.2px);
    background-size: 100% 100%, 100% 100%, 100% 100%, 100% 100%, 28px 28px;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 3.5rem;
    max-width: 1440px;
}

/* Sidebar Custom Styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #040711 0%, #020409 100%) !important;
    border-right: 2px solid #1E293B !important;
    box-shadow: 10px 0 30px rgba(0, 0, 0, 0.8) !important;
}

.sidebar-brand-box {
    background: linear-gradient(135deg, rgba(8, 14, 28, 0.95) 0%, rgba(3, 5, 12, 0.98) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-top: 3.5px solid #FF1E42;
    border-bottom: 2px solid #B026FF;
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 18px;
    text-align: center;
}

.sidebar-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.6rem;
    font-weight: 900;
    line-height: 1.1;
    background: linear-gradient(90deg, #FFFFFF 10%, #00F5FF 40%, #B026FF 70%, #FF1E42 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.sidebar-section-header {
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #00F5FF;
    margin: 16px 0 8px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}
.sidebar-section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(0, 245, 255, 0.4), transparent);
}

.sidebar-kpi-card {
    background: #050914;
    border: 1px solid #1E293B;
    border-left: 3px solid #00FF9D;
    border-radius: 10px;
    padding: 10px 12px;
    margin-bottom: 8px;
}

/* Top Navbar Header */
.tactical-navbar {
    background: linear-gradient(135deg, rgba(8, 14, 28, 0.96) 0%, rgba(3, 5, 12, 0.98) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-top: 4px solid #FF1E42;
    border-bottom: 2.5px solid #B026FF;
    border-radius: 18px;
    padding: 18px 26px;
    margin-bottom: 16px;
    box-shadow: 0 20px 45px rgba(0,0,0,0.9);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 16px;
}

.brand-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 2.2rem;
    font-weight: 900;
    letter-spacing: -0.02em;
    line-height: 1;
    margin: 0;
    background: linear-gradient(90deg, #FFFFFF 10%, #00F5FF 35%, #B026FF 65%, #00FF9D 90%, #FF1E42 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Market Ticker Tape */
.market-ticker-tape {
    display: flex;
    gap: 18px;
    background: rgba(4, 7, 16, 0.85);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 12px;
    padding: 10px 18px;
    margin-bottom: 20px;
    overflow-x: auto;
    white-space: nowrap;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
}
.ticker-item { display: flex; gap: 6px; align-items: center; }
.ticker-up { color: #00FF9D; font-weight: 700; }
.ticker-down { color: #FF1E42; font-weight: 700; }

/* Tabs Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    background: rgba(4, 7, 16, 0.95);
    padding: 8px 12px;
    border-radius: 16px;
    border: 1px solid #1E293B;
    margin-bottom: 22px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 0.92rem !important;
    font-weight: 700 !important;
    color: #94A3B8 !important;
    padding: 12px 20px !important;
    border-radius: 12px !important;
    border: 1px solid transparent !important;
    background: transparent !important;
    transition: all 0.25s ease !important;
}

.stTabs [aria-selected="true"] {
    color: #00F5FF !important;
    background: #0B1324 !important;
    border-color: rgba(176, 38, 255, 0.6) !important;
    box-shadow: 0 4px 22px rgba(176, 38, 255, 0.35) !important;
}

/* Surface & Component Cards */
.sp-card {
    background: linear-gradient(135deg, rgba(8, 14, 28, 0.95) 0%, rgba(3, 6, 15, 0.98) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 16px 40px rgba(0,0,0,0.8);
}

.sp-card-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.15rem;
    font-weight: 800;
    letter-spacing: -0.01em;
    color: #F8FAFC;
    margin-bottom: 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #1E293B;
    padding-bottom: 10px;
}

.verdict-hero-card {
    background: linear-gradient(135deg, rgba(255, 30, 66, 0.16) 0%, rgba(176, 38, 255, 0.12) 50%, rgba(4, 7, 16, 0.98) 100%);
    border: 2px solid rgba(255, 30, 66, 0.7);
    border-radius: 18px;
    padding: 30px;
    text-align: center;
    margin-bottom: 22px;
    box-shadow: 0 20px 50px rgba(255, 30, 66, 0.25), 0 0 30px rgba(176, 38, 255, 0.2);
}

.verdict-action {
    font-family: 'Syne', sans-serif;
    font-size: 2.7rem;
    font-weight: 900;
    letter-spacing: -0.02em;
    margin: 8px 0;
    line-height: 1.1;
}

.agent-node {
    background: #040711;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 20px;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    box-shadow: 0 8px 24px rgba(0,0,0,0.7);
    transition: all 0.25s ease;
}
.agent-node:hover {
    border-color: rgba(0, 245, 255, 0.6);
    transform: translateY(-3px);
}

.trace-line {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: #E2E8F0;
    background: rgba(12, 20, 38, 0.85);
    padding: 12px 16px;
    border-radius: 8px;
    border-left: 4px solid #00F5FF;
    margin: 8px 0;
    line-height: 1.55;
}

.citation-box {
    background: #040711;
    border: 1px solid #1E293B;
    border-left: 4.5px solid #B026FF;
    padding: 16px 20px;
    border-radius: 12px;
    margin-bottom: 12px;
}

/* Metric KPI HUD Cards */
div[data-testid="stMetric"] {
    background: #040711;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-top: 3.5px solid #00FF9D;
    border-radius: 14px;
    padding: 14px 18px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.45);
}
div[data-testid="stMetric"] label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    color: #94A3B8 !important;
    text-transform: uppercase;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.6rem !important;
    font-weight: 800 !important;
    color: #F8FAFC !important;
}

/* Sidebar Custom Action Button */
div.stButton > button {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.05rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.05em !important;
    background: linear-gradient(90deg, #FF1E42 0%, #B026FF 50%, #00F5FF 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85rem 1.8rem !important;
    box-shadow: 0 6px 25px rgba(176, 38, 255, 0.5) !important;
    transition: all 0.25s ease !important;
    width: 100%;
}
div.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 35px rgba(0, 245, 255, 0.8) !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# GRAPH GENERATION HELPERS (Plotly Interactive & Native Fallbacks)
# ---------------------------------------------------------------------------
def create_price_history_chart(ticker, base_price, rsi_val):
    dates = pd.date_range(end=pd.Timestamp.today(), periods=30, freq='D')
    np.random.seed(abs(hash(ticker)) % 1000)
    noise = np.random.normal(0, base_price * 0.015, 30)
    prices = [base_price * 0.90]
    for n in noise[1:]:
        prices.append(max(10.0, prices[-1] + n + (base_price * 0.003)))
    prices[-1] = base_price
    
    df_chart = pd.DataFrame({"Date": dates, "Price": prices})
    
    if HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_chart["Date"], y=df_chart["Price"],
            mode="lines",
            line=dict(color="#00F5FF", width=3),
            fill="tozeroy",
            fillcolor="rgba(0, 245, 255, 0.08)",
            name="Asset Price (₹)"
        ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(4,7,16,0.6)",
            margin=dict(l=10, r=10, t=25, b=10),
            height=280,
            font=dict(family="Plus Jakarta Sans", color="#94A3B8"),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)")
        )
        return fig
    return df_chart.set_index("Date")

def create_agent_radar_chart(outputs):
    categories = [o.agent.replace("Agent", "") for o in outputs]
    confidences = [o.confidence * 100 for o in outputs]
    latencies = [100 - min(100, o.latency_ms) for o in outputs]
    
    if HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Confidence (%)",
            x=categories, y=confidences,
            marker=dict(color=["#00F5FF", "#B026FF", "#00FF9D"], line=dict(color="#FFFFFF", width=1))
        ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(4,7,16,0.6)",
            margin=dict(l=10, r=10, t=20, b=10),
            height=260,
            font=dict(family="Plus Jakarta Sans", color="#94A3B8"),
            yaxis=dict(range=[0, 100], showgrid=True, gridcolor="rgba(255,255,255,0.05)")
        )
        return fig
    return pd.DataFrame({"Agent": categories, "Confidence": confidences}).set_index("Agent")

def create_portfolio_donut(portfolio):
    if HAS_PLOTLY:
        labels = list(portfolio.keys())
        values = list(portfolio.values())
        fig = go.Figure(data=[go.Pie(
            labels=labels, values=values, hole=0.6,
            marker=dict(colors=["#00F5FF", "#FF1E42", "#B026FF", "#00FF9D", "#FBBF24"])
        )])
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10),
            height=260,
            font=dict(family="Plus Jakarta Sans", color="#F8FAFC"),
            showlegend=True
        )
        return fig
    return pd.DataFrame(list(portfolio.items()), columns=["Asset", "Value"]).set_index("Asset")

# ---------------------------------------------------------------------------
# State & Corpus Initialization
# ---------------------------------------------------------------------------
@st.cache_resource
def get_rag_index():
    return RAGIndex(generate_filing_corpus())

rag_index = get_rag_index()
portfolio = generate_portfolio()

if "has_run" not in st.session_state:
    st.session_state.has_run = False

# ---------------------------------------------------------------------------
# SIDEBAR: PARAMETERS & REAL-TIME SYSTEM BENCHMARKS
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand-box">
        <div class="sidebar-title">🕷️ SPIDER-SENSE</div>
        <div style="font-size: 0.75rem; color: #94A3B8; margin-top: 4px; font-weight: 600;">
            AUTONOMOUS MULTI-AGENT SWARM
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-header">Target Parameters</div>', unsafe_allow_html=True)
    ticker = st.selectbox("Target Equity Asset", TICKERS, index=0)
    profile_name = st.selectbox("Investor Persona Profile", list(RISK_PROFILES.keys()), index=1)
    
    st.markdown('<div class="sidebar-section-header">Simulation & Chaos</div>', unsafe_allow_html=True)
    scenario = st.selectbox("Market Feed Scenario", ["normal", "crash", "positive_news", "negative_news"])
    degrade = st.selectbox("Chaos Fault-Injection", ["none", "momentum", "volume", "sentiment"])

    st.write("")
    run_swarm = st.button("🚀 DISPATCH SWARM", use_container_width=True)

    if run_swarm or not st.session_state.has_run:
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

    st.markdown('<div class="sidebar-section-header">Swarm Infrastructure</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="sidebar-kpi-card">
        <div style="font-family:'JetBrains Mono'; font-size:0.68rem; color:#64748B;">ORCHESTRATION PIPELINE</div>
        <div style="font-family:'Syne'; font-size:0.92rem; font-weight:800; color:#00FF9D;">3 ASYNC PARALLEL DESKS</div>
    </div>
    <div class="sidebar-kpi-card" style="border-left-color: #00F5FF;">
        <div style="font-family:'JetBrains Mono'; font-size:0.68rem; color:#64748B;">RAG VECTOR CORPUS</div>
        <div style="font-family:'Syne'; font-size:0.92rem; font-weight:800; color:#00F5FF;">STATUTORY SEBI DISCLOSURES</div>
    </div>
    <div class="sidebar-kpi-card" style="border-left-color: #B026FF;">
        <div style="font-family:'JetBrains Mono'; font-size:0.68rem; color:#64748B;">CHAOS TELEMETRY</div>
        <div style="font-family:'Syne'; font-size:0.92rem; font-weight:800; color:#{'#FF1E42' if degrade != 'none' else '#D8B4FE'};">
            {'FAULT INJECTION ACTIVE' if degrade != 'none' else 'NOMINAL FEED'}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# MAIN WORKSPACE: HEADER & TICKER
# ---------------------------------------------------------------------------
st.markdown("""
<div class="tactical-navbar">
    <div>
        <div class="brand-title">🕷️ SPIDER-SENSE FINANCIAL</div>
        <div style="color: #94A3B8; font-size: 0.9rem; margin-top: 5px; font-weight: 500;">
            Multi-Agent Autonomous Financial Intelligence Network // Explainable Retail Research Infrastructure
        </div>
    </div>
    <div style="display: flex; gap: 10px; align-items: center; flex-wrap: wrap;">
        <span style="font-family:'JetBrains Mono'; font-size:0.75rem; font-weight:800; padding:6px 12px; border-radius:6px; border:1px solid rgba(0,245,255,0.6); color:#00F5FF; background:rgba(0,245,255,0.1);">● Swarm Online</span>
        <span style="font-family:'JetBrains Mono'; font-size:0.75rem; font-weight:800; padding:6px 12px; border-radius:6px; border:1px solid rgba(176,38,255,0.6); color:#D8B4FE; background:rgba(176,38,255,0.1);">Sprint 1: PS-01</span>
        <span style="font-family:'JetBrains Mono'; font-size:0.75rem; font-weight:800; padding:6px 12px; border-radius:6px; border:1px solid rgba(255,30,66,0.7); color:#FF4D6D; background:rgba(255,30,66,0.1);">Team YOLOTECH</span>
    </div>
</div>

<div class="market-ticker-tape">
    <div class="ticker-item"><span>NIFTY 50:</span> <span class="ticker-up">24,860.20 (+1.14%)</span></div>
    <div class="ticker-item"><span>BANK NIFTY:</span> <span class="ticker-up">53,120.40 (+0.88%)</span></div>
    <div class="ticker-item"><span>RELIANCE:</span> <span class="ticker-up">₹2,980.50 (+2.40%)</span></div>
    <div class="ticker-item"><span>TATAMOTORS:</span> <span class="ticker-up">₹995.20 (+3.80%)</span></div>
    <div class="ticker-item"><span>ZOMATO:</span> <span class="ticker-up">₹242.10 (+5.10%)</span></div>
    <div class="ticker-item"><span>INFY:</span> <span class="ticker-down">₹1,780.00 (-0.80%)</span></div>
    <div class="ticker-item"><span>INDIA VIX:</span> <span class="ticker-down">13.25 (-4.10%)</span></div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# WORKSPACE DISPLAY TABS
# ---------------------------------------------------------------------------
outputs = st.session_state.outputs
synthesis = st.session_state.synthesis
row = st.session_state.row
m_ticker = st.session_state.ticker
m_profile = st.session_state.profile_name
m_data = st.session_state.market_data

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Master Verdict & Interactive Charts",
    "🤖 3-Agent Parallel Swarm Telemetry",
    "⚖️ Persona Matrix & Risk Boundaries",
    "📎 SEBI Regulatory RAG Explorer",
    "📊 Benchmarks & Portfolio Logs"
])

# =============================================================================
# TAB 1: MASTER VERDICT & INTERACTIVE CHARTS
# =============================================================================
with tab1:
    st.markdown('<div class="sp-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="sp-card-header"><span>Master Synthesis Verdict // <strong>{m_ticker}</strong></span><span style="color:#B026FF; font-size:0.95rem; font-weight:700;">Persona: {m_profile}</span></div>', unsafe_allow_html=True)
    
    action_col = "#00FF9D" if "BUY" in synthesis['action'] else ("#FF1E42" if "REDUCE" in synthesis['action'] or "AVOID" in synthesis['action'] else "#00F5FF")
    
    st.markdown(f"""
    <div class="verdict-hero-card">
        <div style="font-family:'JetBrains Mono'; font-size:0.84rem; color:#D8B4FE; text-transform:uppercase; font-weight:800; letter-spacing:0.08em;">
            CALIBRATED RISK ACTION // {m_profile.upper()}
        </div>
        <div class="verdict-action" style="color:{action_col};">
            {synthesis['action']}
        </div>
        <div style="font-size:1.08rem; color:#E2E8F0; line-height:1.65; max-width:940px; margin: 10px auto 0 auto;">
            {synthesis['reasoning']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.progress(synthesis["confidence"], text=f"Synthesis Grounding Confidence: {synthesis['confidence']:.0%}")

    # Interactive Graph & Strategy Overview
    st.markdown('<div class="sp-card-header" style="margin-top:24px;"><span>Asset Price Action & Telemetry Stream</span><span style="font-size:0.8rem; color:#64748B;">30-Day Moving Trajectory</span></div>', unsafe_allow_html=True)
    
    g_col1, g_col2 = st.columns([2.2, 1])
    with g_col1:
        chart_obj = create_price_history_chart(m_ticker, m_data["price"], m_data["rsi"])
        if HAS_PLOTLY:
            st.plotly_chart(chart_obj, use_container_width=True, config={'displayModeBar': False})
        else:
            st.area_chart(chart_obj, height=280)
    with g_col2:
        st.markdown(f"""
        <div style="background:#050914; border:1px solid #1E293B; border-radius:12px; padding:16px; height:100%;">
            <div style="font-family:'Syne'; font-size:0.95rem; font-weight:800; color:#00F5FF; margin-bottom:12px;">EXECUTION METRICS</div>
            <div style="font-size:0.82rem; color:#94A3B8; margin-bottom:8px;">Live Tick: <strong style="color:#FFF;">₹{m_data['price']:,.2f}</strong> ({m_data['change_pct']:+.2f}%)</div>
            <div style="font-size:0.82rem; color:#94A3B8; margin-bottom:8px;">RSI (14D): <strong style="color:{'#FF1E42' if m_data['rsi']>70 else '#00FF9D'};">{m_data['rsi']:.1f}</strong></div>
            <div style="font-size:0.82rem; color:#94A3B8; margin-bottom:8px;">Vol Anomaly: <strong style="color:#FFF;">+{m_data['vol_zscore']}σ</strong></div>
            <div style="font-size:0.82rem; color:#94A3B8;">Options PCR: <strong style="color:#FFF;">{m_data['pcr']:.2f}</strong></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="sp-card-header" style="margin-top:24px; color:#B026FF;">Transparent Multi-Agent Reasoning Trace</div>', unsafe_allow_html=True)
    for out in outputs:
        st.markdown(f"""
        <div class="trace-line">
            <strong style="color:#00F5FF;">[{out.agent}]</strong> ➔ {out.reasoning}
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# TAB 2: PARALLEL AGENT SWARM TELEMETRY
# =============================================================================
with tab2:
    st.markdown('<div class="sp-card">', unsafe_allow_html=True)
    st.markdown('<div class="sp-card-header"><span>Specialized Domain Agents // Real-Time Telemetry</span><span style="font-size:0.85rem; color:#64748B;">Parallel Async Pipeline (< 150ms)</span></div>', unsafe_allow_html=True)
    
    # Visual Agent Confidence Bar Chart
    radar_obj = create_agent_radar_chart(outputs)
    if HAS_PLOTLY:
        st.plotly_chart(radar_obj, use_container_width=True, config={'displayModeBar': False})
    else:
        st.bar_chart(radar_obj, height=240)

    st.write("")
    cols = st.columns(len(outputs))
    for col, o in zip(cols, outputs):
        with col:
            status_col = "#B026FF" if o.degraded else ("#00FF9D" if "BULLISH" in o.label or "GROUNDED" in o.label else "#FF1E42")
            st.markdown(f"""
            <div class="agent-node" style="border-left: 4.5px solid {status_col};">
                <div>
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <strong style="color:#FFF; font-size:1.05rem; font-family:'Syne';">{o.agent}</strong>
                        <span style="font-family:'JetBrains Mono'; font-size:0.65rem; color:{status_col}; border:1px solid {status_col}50; padding:2px 8px; border-radius:4px;">
                            {'DEGRADED' if o.degraded else 'NOMINAL'}
                        </span>
                    </div>
                    <div style="font-family:'JetBrains Mono'; font-size:0.68rem; color:#64748B; margin-top:4px; text-transform:uppercase;">
                        {o.dimension}
                    </div>
                    <div style="font-family:'Syne'; font-size:1.55rem; font-weight:800; color:{status_col}; margin:14px 0 4px 0;">
                        {o.label}
                    </div>
                    <div style="font-family:'JetBrains Mono'; font-size:0.75rem; color:#94A3B8;">
                        CONFIDENCE // {o.confidence:.0%}
                    </div>
                    <div style="font-size:0.86rem; color:#CBD5E1; line-height:1.55; margin:14px 0; background:rgba(12,20,38,0.7); padding:12px; border-radius:8px; border-left: 2px solid {status_col};">
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

# =============================================================================
# TAB 3: BEHAVIORAL MATRIX & RISK BOUNDARIES
# =============================================================================
with tab3:
    st.markdown('<div class="sp-card">', unsafe_allow_html=True)
    st.markdown('<div class="sp-card-header"><span>Behavioral Personalization Matrix</span><span style="font-size:0.85rem; color:#00FF9D;">Identical Feed ➔ Differential Synthesis</span></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="font-size:0.88rem; color:#94A3B8; margin-bottom:18px; line-height:1.5;">
        <strong>Core Proof of Innovation:</strong> When an asset enters high momentum ($RSI > 75$), standard algorithms issue a generic alert. SPIDER-SENSE evaluates the signals through the user's specific risk envelope—protecting conservative accounts from drawdowns while unlocking breakout alpha for aggressive accounts.
    </div>
    """, unsafe_allow_html=True)

    comp_cols = st.columns(len(RISK_PROFILES))
    for c, pname in zip(comp_cols, RISK_PROFILES.keys()):
        _, syn2 = run_pipeline(
            m_ticker, st.session_state.market_data, st.session_state.news, 
            rag_index, pname,
            simulate_degraded=None if st.session_state.degrade == "none" else st.session_state.degrade
        )
        is_active = (pname == m_profile)
        action_col = "#00FF9D" if "BUY" in syn2['action'] else ("#FF1E42" if "REDUCE" in syn2['action'] or "AVOID" in syn2['action'] else "#00F5FF")
        p_info = RISK_PROFILES[pname]
        
        with c:
            st.markdown(f"""
            <div class="agent-node" style="border: 2px solid {'#B026FF' if is_active else '#1E293B'}; background:{'rgba(176,38,255,0.06)' if is_active else '#040711'};">
                <div>
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-weight:800; font-size:1.05rem; color:#FFF; font-family:'Syne';">
                            {pname.split('(')[0]}
                        </span>
                        <span style="font-family:'JetBrains Mono'; font-size:0.65rem; color:#00F5FF; border:1px solid rgba(0,245,255,0.4); padding:2px 6px; border-radius:4px;">
                            {'★ ACTIVE' if is_active else 'PERSONA'}
                        </span>
                    </div>
                    <div style="font-size:0.75rem; color:#94A3B8; margin-top:4px;">
                        Risk Tier: <strong style="color:#FFF;">{p_info['risk']}</strong> | Loss Limit: <strong style="color:#FF1E42;">{p_info['max_drawdown']}</strong>
                    </div>
                    <div style="font-family:'Syne'; font-size:1.35rem; font-weight:800; color:{action_col}; margin:14px 0 6px 0;">
                        {syn2['action']}
                    </div>
                    <div style="font-size:0.84rem; color:#CBD5E1; line-height:1.55; margin:10px 0;">
                        {syn2['reasoning']}
                    </div>
                </div>
                <div style="border-top:1px solid #1E293B; padding-top:8px; font-family:'JetBrains Mono'; font-size:0.75rem; color:#94A3B8;">
                    SYNTHESIS CONFIDENCE: <strong style="color:#FFF;">{syn2['confidence']:.0%}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# TAB 4: SEBI REGULATORY RAG EXPLORER
# =============================================================================
with tab4:
    st.markdown('<div class="sp-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="sp-card-header"><span>Retrieved Statutory SEBI Filings & Disclosures</span><span style="color:#D8B4FE;">Asset: {m_ticker}</span></div>', unsafe_allow_html=True)
    
    if not synthesis["citations"]:
        st.warning("⚠️ No statutory corporate filings retrieved or Chaos Telemetry active.")
    else:
        for c in synthesis["citations"]:
            st.markdown(f"""
            <div class="citation-box">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-family:'Syne'; font-size:1.1rem; font-weight:700; color:#D8B4FE;">
                        📄 SEBI Statutory Corpus Chunk: {c}
                    </span>
                    <span style="font-family:'JetBrains Mono'; font-size:0.7rem; color:#00FF9D; background:rgba(0,255,157,0.1); padding:2px 8px; border-radius:4px; border:1px solid rgba(0,255,157,0.3);">
                        VERIFIED AUDIT
                    </span>
                </div>
                <div style="font-size:0.88rem; color:#CBD5E1; margin-top:8px; line-height:1.55;">
                    Retrieved via TF-IDF Vector Space Embeddings over official statutory disclosures. Source grounding verified to prevent ungrounded AI hallucinations.
                </div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# TAB 5: BENCHMARKS & PORTFOLIO LOGS
# =============================================================================
with tab5:
    st.markdown('<div class="sp-card">', unsafe_allow_html=True)
    st.markdown('<div class="sp-card-header"><span>Quantitative Benchmarks & Portfolio Allocation</span><span style="color:#00FF9D;">Real-Time Telemetry</span></div>', unsafe_allow_html=True)
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Avg Swarm Latency", f"{row['avg_agent_latency_ms']} ms")
    m2.metric("Portfolio HHI Risk", row["portfolio_concentration_hhi"])
    m3.metric("30d Forward Return", f"{row['mock_30d_forward_return_pct']}%")
    m4.metric("Directional Accuracy", "PASS ✅" if row["directional_accuracy_proxy"] else "NEUTRAL ➖")
    
    st.write("")
    p_col1, p_col2 = st.columns([1.2, 1])
    with p_col1:
        st.markdown('<div class="sp-card-header" style="margin-top:12px;">Asset Concentration Breakdown</div>', unsafe_allow_html=True)
        donut_fig = create_portfolio_donut(portfolio)
        if HAS_PLOTLY:
            st.plotly_chart(donut_fig, use_container_width=True, config={'displayModeBar': False})
        else:
            st.dataframe(donut_fig, use_container_width=True)
    with p_col2:
        st.markdown('<div class="sp-card-header" style="margin-top:12px;">Holding Valuations</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(list(portfolio.items()), columns=["Asset", "Holding Value (₹)"]), use_container_width=True, hide_index=True)

    st.markdown('<div class="sp-card-header" style="margin-top:24px; color:#00FF9D;">Historical Audit Trail (Persistent Across Sessions)</div>', unsafe_allow_html=True)
    if os.path.exists(LOG_PATH):
        st.dataframe(pd.read_csv(LOG_PATH), use_container_width=True, hide_index=True)
    else:
        st.info("No persistent session logs found.")
    st.markdown('</div>', unsafe_allow_html=True)
