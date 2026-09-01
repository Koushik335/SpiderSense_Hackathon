import os
import random

import pandas as pd
import streamlit as st

from pipeline import RAGIndex, RISK_PROFILES, LOG_PATH, run_pipeline, log_session
from synthetic_data import TICKERS, generate_market_data, generate_news, generate_filing_corpus, generate_portfolio

st.set_page_config(page_title="Multi-Agent Retail Investment Intelligence", layout="wide")
st.title("📊 Multi-Agent Autonomous Financial Intelligence System")
st.caption("PS-01 · HackVerse Sprint 1 — Explainable, personalized investment intelligence for retail investors")


@st.cache_resource
def get_rag_index():
    return RAGIndex(generate_filing_corpus())


rag_index = get_rag_index()
portfolio = generate_portfolio()

col1, col2, col3 = st.columns(3)
with col1:
    ticker = st.selectbox("Ticker", TICKERS)
with col2:
    profile_name = st.selectbox("User risk profile", list(RISK_PROFILES.keys()), index=1)
with col3:
    scenario = st.selectbox("Market scenario", ["normal", "crash", "positive_news", "negative_news"])
degrade = st.selectbox(
    "Simulate a degraded data feed (for the required 'graceful degradation' demo)",
    ["none", "momentum", "volume", "sentiment"],
)

run = st.button("▶ Run multi-agent analysis", type="primary")

st.subheader("📁 Current Portfolio / Watchlist")
st.dataframe(pd.DataFrame(list(portfolio.items()), columns=["Ticker", "Value (₹)"]), use_container_width=True)

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

    st.subheader("🧠 Agent Signals (parallel execution, structured outputs)")
    cols = st.columns(len(outputs))
    for c, o in zip(cols, outputs):
        with c:
            badge = "🔴 DEGRADED" if o.degraded else "🟢 OK"
            st.markdown(f"**{o.agent}** — {badge}")
            st.metric(o.dimension, o.label, f"conf {o.confidence}")
            st.caption(o.reasoning)
            st.caption(f"latency: {o.latency_ms:.1f} ms")
            if o.citations:
                st.caption("Sources: " + ", ".join(o.citations))

    st.subheader("🎯 Synthesized Recommendation")
    conflict_flag = " ⚠️ CONFLICTING SIGNALS" if synthesis["conflict"] else ""
    st.markdown(f"### {synthesis['action']}{conflict_flag}")
    st.write(synthesis["reasoning"])
    st.progress(synthesis["confidence"])
    if synthesis["citations"]:
        st.caption("📎 Cited sources: " + ", ".join(synthesis["citations"]))
    if synthesis["degraded_agents"]:
        st.warning(
            f"Degraded inputs: {', '.join(synthesis['degraded_agents'])} — recommendation "
            f"issued with reduced confidence, not fabricated."
        )

    st.subheader("👤 Same market data, other risk profiles (proof of personalization)")
    comp_cols = st.columns(len(RISK_PROFILES))
    for c, pname in zip(comp_cols, RISK_PROFILES.keys()):
        _, syn2 = run_pipeline(
            ticker, market_data, news, rag_index, pname,
            simulate_degraded=None if degrade == "none" else degrade,
        )
        with c:
            st.markdown(f"**{pname}**")
            st.write(syn2["action"])
            st.caption(f"confidence {syn2['confidence']}")

    forward_return_mock = random.uniform(-8, 8)
    row = log_session(ticker, profile_name, outputs, synthesis, forward_return_mock, portfolio)
    st.subheader("📈 Performance Log (this session)")
    st.json(row)

st.subheader("🗂 Historical Performance Log (across sessions)")
if os.path.exists(LOG_PATH):
    st.dataframe(pd.read_csv(LOG_PATH), use_container_width=True)
else:
    st.info("Run an analysis above to start logging.")