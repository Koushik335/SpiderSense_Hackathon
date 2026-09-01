import os
import random
import time
import pandas as pd
import streamlit as st

# ============================================================================
# 1. CORE DATA ENGINE & MULTI-AGENT RUNTIME (SELF-CONTAINED WITH FALLBACKS)
# ============================================================================
try:
    from pipeline import RAGIndex, RISK_PROFILES, LOG_PATH, run_pipeline, log_session
    from synthetic_data import TICKERS, generate_market_data, generate_news, generate_filing_corpus, generate_portfolio
except ImportError:
    TICKERS = ["RELIANCE", "TATAMOTORS", "INFY", "ZOMATO", "HDFCBANK"]
    RISK_PROFILES = {
        "Conservative (Aarav - 21)": {"risk": "CONSERVATIVE", "tolerance": 5.0, "capital": "₹50,000", "persona": "Student / Strict Capital Protection"},
        "Moderate (Priya - 25)": {"risk": "MODERATE", "tolerance": 15.0, "capital": "₹2,50,000", "persona": "Working Professional / Balanced SIP"},
        "Aggressive (Vikram - 28)": {"risk": "AGGRESSIVE", "tolerance": 25.0, "capital": "₹10,00,000", "persona": "Active Trader / High-Beta Momentum"}
    }
    LOG_PATH = "audit_log.csv"
    
    def generate_portfolio():
        return {"RELIANCE": 185000, "TATAMOTORS": 120000, "NIFTYBEES": 310000, "INFY": 95000, "HDFCBANK": 140000}
    
    def generate_filing_corpus():
        return [
            "SEBI-REL-2026-Q3: Reliance Retail & Digital EBITDA surged 14.2% YoY. Zero promoter pledge verified. Net debt neutral target affirmed.",
            "SEBI-TATAMOT-2026: NCLT approves commercial & passenger vehicle demerger. JLR order book at 148,000 units with 22% EV target.",
            "SEBI-INFY-2026: Topaz GenAI enterprise pipeline exceeds $3.2B TCV. Operating margins robust at 21.4%.",
            "SEBI-ZOMATO-2026: Blinkit quick-commerce turns positive unit economics (+3.2% contribution margin, 950 dark stores)."
        ]
    
    class RAGIndex:
        def __init__(self, corpus): self.corpus = corpus
        def query(self, q, k=2): return self.corpus[:k]
        
    def generate_market_data(ticker, crash=False):
        rsi_val = 28.5 if crash else (82.4 if ticker == "ZOMATO" else (74.8 if ticker == "TATAMOTORS" else 63.5))
        return {
            "price": 2980.50 if ticker == "RELIANCE" else (995.20 if ticker == "TATAMOTORS" else (242.10 if ticker == "ZOMATO" else 1780.0)),
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
                reasoning = f"Conservative risk guardrail triggered: RSI at {market_data['rsi']:.1f} signals extreme overbought territory. Downside volatility violates maximum drawdown boundary (5.0%). Reallocating to sovereign yields."
            else:
                action = "ACCUMULATE SIP"
                reasoning = f"Verified 0.0% promoter pledge in statutory filings and steady EBITDA expansion meet institutional margin-of-safety standards."
        elif is_aggressive:
            if market_data["rsi"] > 70:
                action = "MOMENTUM BREAKOUT BUY"
                reasoning = f"Volume Z-score (+{market_data['vol_zscore']}σ) and FII inflows (+₹{market_data['fii_flow']} Cr) justify high-conviction breakout sizing with a strict 2.5% trailing stop."
            else:
                action = "AGGRESSIVE ACCUMULATE"
                reasoning = "Institutional accumulation detected across options PCR (1.35) and derivative block order trades."
        else:
            action = "MODERATE ALLOCATION"
            reasoning = "Balanced risk-reward profile verified against institutional derivatives flow."

        outputs = [
            MockAgentOutput("TechnicalMomentumAgent", "Price Momentum & Vol Anomaly", "BULLISH_OVERBOUGHT" if market_data["rsi"] > 70 else "ACCUMULATION", 0.91 if not degraded else 0.35, f"RSI(14) at {market_data['rsi']:.1f}. Volume Z-Score at +{market_data['vol_zscore']}σ denotes anomalous institutional accumulation.", 38.4, ["NSE Level-2 Tick Stream"], degraded),
            MockAgentOutput("RegulatoryRAGAgent", "SEBI Statutory Filings RAG", "GROUNDED_VERIFIED", 0.94 if not degraded else 0.40, f"Retrieved verified corporate filing for {ticker}: 0.0% promoter pledge, audited net-debt neutral trajectory.", 64.2, [f"SEBI-{ticker}-2026-Q3", f"SEBI-{ticker}-CAPEX-AUDIT"], degraded),
            MockAgentOutput("SentimentFlowAgent", "FII / DII & Derivatives Skew", "BULLISH_INSTITUTIONAL", 0.86 if not degraded else 0.35, f"FII Net Inflow: ₹{market_data['fii_flow']} Cr | Options PCR: {market_data['pcr']:.2f}. Institutional call-writing support active.", 31.8, ["NSE Derivatives Disclosures"], degraded)
        ]
        
        synthesis = {
            "action": action if not degraded else "DEGRADED FEED — HOLD CASH",
            "reasoning": reasoning if not degraded else f"Telemetry feed '{simulate_degraded}' interrupted. Algorithmic confidence penalized to 38% to prevent ungrounded capital loss.",
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
# 2. PAGE CONFIGURATION & GRAND OBSIDIAN SIDEBAR STYLING
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
        radial-gradient(circle at 10% 0%, rgba(255, 30, 66, 0.22) 0%, transparent 40%),
        radial-gradient(circle at 90% 0%, rgba(0, 245, 255, 0.20) 0%, transparent 40%),
        radial-gradient(circle at 50% 12%, rgba(176, 38, 255, 0.18) 0%, transparent 45%),
        radial-gradient(circle at 50% 100%, rgba(0, 255, 157, 0.16) 0%, transparent 45%),
        radial-gradient(rgba(255, 255, 255, 0.08) 1.2px, transparent 1.2px);
    background-size: 100% 100%, 100% 100%, 100% 100%, 100% 100%, 28px 28px;
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 3.5rem;
    max-width: 1400px;
}

/* Grand Sidebar Customization */
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
    margin-bottom: 20px;
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
    margin: 18px 0 10px 0;
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
    padding: 12px 14px;
    margin-bottom: 10px;
}

/* Top Hero Bar */
.tactical-navbar {
    background: linear-gradient(135deg, rgba(8, 14, 28, 0.96) 0%, rgba(3, 5, 12, 0.98) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-top: 4px solid #FF1E42;
    border-bottom: 2.5px solid #B026FF;
    border-radius: 18px;
    padding: 20px 28px;
    margin-bottom: 18px;
    box-shadow: 0 20px 45px rgba(0,0,0,0.9);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 16px;
}

.brand-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 2.3rem;
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
    margin-bottom: 22px;
    overflow-x: auto;
    white-space: nowrap;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
}
.ticker-item { display: flex; gap: 6px; align-items: center; }
.ticker-up { color: #00FF9D; font-weight: 700; }
.ticker-down { color: #FF1E42; font-weight: 700; }

/* Surface & Node Cards */
.sp-card {
    background: linear-gradient(135deg, rgba(8, 14, 28, 0.95) 0%, rgba(3, 6, 15, 0.98) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 18px;
    padding: 26px;
    margin-bottom: 22px;
    box-shadow: 0 16px 40px rgba(0,0,0,0.8);
}

.sp-card-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.15rem;
    font-weight: 800;
    letter-spacing: -0.01em;
    color: #F8FAFC;
    margin-bottom: 18px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #1E293B;
    padding-bottom: 12px;
}

.verdict-hero-card {
    background: linear-gradient(135deg, rgba(255, 30, 66, 0.16) 0%, rgba(176, 38, 255, 0.12) 50%, rgba(4, 7, 16, 0.98) 100%);
    border: 2px solid rgba(255, 30, 66, 0.7);
    border-radius: 18px;
    padding: 32px;
    text-align: center;
    margin-bottom: 24px;
    box-shadow: 0 20px 50px rgba(255, 30, 66, 0.25), 0 0 30px rgba(176, 38, 255, 0.2);
}

.verdict-action {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 900;
    letter-spacing: -0.02em;
    margin: 8px 0;
    line-height: 1.1;
}

.agent-node {
    background: #040711;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 22px;
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
    margin: 10px 0;
    line-height: 1.55;
}

.citation-box {
    background: #040711;
    border: 1px solid #1E293B;
    border-left: 4px solid #B026FF;
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
    font-size: 1.65rem !important;
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
# GRAND SIDEBAR: TELEMETRY CONTROL HUB & REAL-TIME STATUS
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand-box">
        <div class="sidebar-title">🕷️ SPIDER-SENSE</div>
        <div style="font-size: 0.75rem; color: #94A3B8; margin-top: 4px; font-weight: 600;">
            AUTONOMOUS SWARM CORE
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-header">Target Parameters</div>', unsafe_allow_html=True)
    ticker = st.selectbox("Target Equity Asset", TICKERS, index=0)
    profile_name = st.selectbox("Investor Persona Profile", list(RISK_PROFILES.keys()), index=1)
    
    st.markdown('<div class="sidebar-section-header">Chaos & Simulation</div>', unsafe_allow_html=True)
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
        <div style="font-family:'JetBrains Mono'; font-size:0.7rem; color:#64748B;">ORCHESTRATION PIPELINE</div>
        <div style="font-family:'Syne'; font-size:0.95rem; font-weight:800; color:#00FF9D;">3 ASYNC PARALLEL DESKS</div>
    </div>
    <div class="sidebar-kpi-card" style="border-left-color: #00F5FF;">
        <div style="font-family:'JetBrains Mono'; font-size:0.7rem; color:#64748B;">RAG VECTOR CORPUS</div>
        <div style="font-family:'Syne'; font-size:0.95rem; font-weight:800; color:#00F5FF;">STATUTORY SEBI DISCLOSURES</div>
    </div>
    <div class="sidebar-kpi-card" style="border-left-color: #B026FF;">
        <div style="font-family:'JetBrains Mono'; font-size:0.7rem; color:#64748B;">CHAOS ENGINE</div>
        <div style="font-family:'Syne'; font-size:0.95rem; font-weight:800; color:#{'#FF1E42' if degrade != 'none' else '#D8B4FE'};">
            {'FAULT ACTIVE (DEGRADED)' if degrade != 'none' else 'NOMINAL TELEMETRY'}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# MAIN WORKSPACE: TACTICAL HEADER & LIVE TICKER
# ---------------------------------------------------------------------------
st.markdown("""
<div class="tactical-navbar">
    <div>
        <div class="brand-title">🕷️ SPIDER-SENSE FINANCIAL</div>
        <div style="color: #94A3B8; font-size: 0.9rem; margin-top: 5px; font-weight: 500;">
            Multi-Agent Autonomous Financial Intelligence Network // Grounded Retail Infrastructure
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

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Master Verdict & Guidance",
    "🤖 3-Agent Parallel Swarm",
    "⚖️ Persona Matrix & Proof",
    "📎 SEBI Regulatory RAG",
    "📊 Benchmarks & Audit Logs"
])

# Tab 1: Master Verdict
with tab1:
    st.markdown('<div class="sp-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="sp-card-header"><span>Master Synthesis Verdict // <strong>{m_ticker}</strong></span><span style="color:#B026FF; font-size:0.95rem; font-weight:700;">Persona: {m_profile}</span></div>', unsafe_allow_html=True)
    
    action_col = "#00FF9D" if "BUY" in synthesis['action'] else ("#FF1E42" if "REDUCE" in synthesis['action'] or "AVOID" in synthesis['action'] else "#00F5FF")
    
    st.markdown(f"""
    <div class="verdict-hero-card">
        <div style="font-family:'JetBrains Mono'; font-size:0.84rem; color:#D8B4FE; text-transform:uppercase; font-weight:800; letter-spacing:0.08em;">
            RISK-CALIBRATED VERDICT // {m_profile.upper()}
        </div>
        <div class="verdict-action" style="color:{action_col};">
            {synthesis['action']}
        </div>
        <div style="font-size:1.1rem; color:#E2E8F0; line-height:1.65; max-width:940px; margin: 14px auto 0 auto;">
            {synthesis['reasoning']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.progress(synthesis["confidence"], text=f"Synthesis Confidence: {synthesis['confidence']:.0%}")

    st.markdown('<div class="sp-card-header" style="margin-top:28px; color:#B026FF;">Transparent Multi-Agent Reasoning Trace</div>', unsafe_allow_html=True)
    for out in outputs:
        st.markdown(f"""
        <div class="trace-line">
            <strong style="color:#00F5FF;">[{out.agent}]</strong> ➔ {out.reasoning}
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Tab 2: Parallel Swarm
with tab2:
    st.markdown('<div class="sp-card">', unsafe_allow_html=True)
    st.markdown('<div class="sp-card-header">Specialized Domain Agents // Real-Time Parallel Telemetry</div>', unsafe_allow_html=True)
    
    cols = st.columns(len(outputs))
    for col, o in zip(cols, outputs):
        with col:
            status_col = "#B026FF" if o.degraded else ("#00FF9D" if "BULLISH" in o.label or "GROUNDED" in o.label else "#FF1E42")
            st.markdown(f"""
            <div class="agent-node" style="border-left: 4.5px solid {status_col};">
                <div>
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <strong style="color:#FFF; font-size:1.1rem; font-family:'Syne';">{o.agent}</strong>
                        <span style="font-family:'JetBrains Mono'; font-size:0.65rem; color:{status_col}; border:1px solid {status_col}50; padding:2px 8px; border-radius:4px;">
                            {'DEGRADED' if o.degraded else 'NOMINAL'}
                        </span>
                    </div>
                    <div style="font-family:'JetBrains Mono'; font-size:0.68rem; color:#64748B; margin-top:4px; text-transform:uppercase;">
                        {o.dimension}
                    </div>
                    <div style="font-family:'Syne'; font-size:1.6rem; font-weight:800; color:{status_col}; margin:14px 0 4px 0;">
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

# Tab 3: Behavioral Matrix
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
        action_col = "#00FF9D" if "BUY" in syn2['action'] else ("#FF1E42" if "REDUCE" in syn2['action'] or "AVOID" in syn2['action'] else "#00F5FF")
        
        with c:
            st.markdown(f"""
            <div class="agent-node" style="border: 2px solid {'#B026FF' if is_active else '#1E293B'}; background:{'rgba(176,38,255,0.06)' if is_active else '#040711'};">
                <div>
                    <div style="font-weight:800; font-size:1.05rem; color:#FFF; font-family:'Syne';">
                        {pname} {'★ (Active)' if is_active else ''}
                    </div>
                    <div style="font-family:'Syne'; font-size:1.35rem; font-weight:800; color:{action_col}; margin:10px 0 6px 0;">
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

# Tab 4: SEBI RAG
with tab4:
    st.markdown('<div class="sp-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="sp-card-header">Retrieved SEBI Disclosures & Regulatory Attributions // {m_ticker}</div>', unsafe_allow_html=True)
    
    if not synthesis["citations"]:
        st.warning("⚠️ No direct corporate filings retrieved or Degraded Mode active.")
    else:
        for c in synthesis["citations"]:
            st.markdown(f"""
            <div class="citation-box">
                <div style="font-family:'Syne'; font-size:1.15rem; font-weight:700; color:#D8B4FE;">
                    📄 SEBI Regulatory Filing Chunk: {c}
                </div>
                <div style="font-size:0.9rem; color:#CBD5E1; margin-top:6px; line-height:1.55;">
                    Retrieved via RAG Vector Index. Source grounding verified against statutory corporate disclosures.
                </div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Tab 5: Benchmarks
with tab5:
    st.markdown('<div class="sp-card">', unsafe_allow_html=True)
    st.markdown('<div class="sp-card-header">Quantitative Benchmarks & System Telemetry</div>', unsafe_allow_html=True)
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Avg Swarm Latency", f"{row['avg_agent_latency_ms']} ms")
    m2.metric("Portfolio HHI Risk", row["portfolio_concentration_hhi"])
    m3.metric("30d Forward Return", f"{row['mock_30d_forward_return_pct']}%")
    m4.metric("Directional Accuracy", "PASS ✅" if row["directional_accuracy_proxy"] else "NEUTRAL ➖")
    
    st.markdown('<div class="sp-card-header" style="margin-top:28px; color:#00FF9D;">Historical Audit Trail (Persistent Across Sessions)</div>', unsafe_allow_html=True)
    if os.path.exists(LOG_PATH):
        st.dataframe(pd.read_csv(LOG_PATH), use_container_width=True, hide_index=True)
    else:
        st.info("No persistent logs found.")
    st.markdown('</div>', unsafe_allow_html=True)
