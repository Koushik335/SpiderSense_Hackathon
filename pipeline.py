"""
Core multi-agent reasoning pipeline.
Framework-agnostic (no LangChain/CrewAI dependency) so it has zero install risk
during the demo. Each agent has a defined role + structured output contract
(AgentOutput), consumed by a synthesis layer. Agents run in parallel via
ThreadPoolExecutor to satisfy the "parallel execution" requirement.
"""
import csv
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Structured output contract every agent must return
# ---------------------------------------------------------------------------
@dataclass
class AgentOutput:
    agent: str
    dimension: str
    label: str
    confidence: float  # 0-1
    reasoning: str
    citations: List[str] = field(default_factory=list)
    degraded: bool = False
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Agent 1: Momentum
# ---------------------------------------------------------------------------
def momentum_agent(market_data: dict) -> AgentOutput:
    t0 = time.time()
    prices = market_data.get("prices")
    if not prices or len(prices) < 5:
        return AgentOutput(
            "MomentumAgent", "price_momentum", "INSUFFICIENT_DATA", 0.0,
            "Price feed missing or too short to compute momentum reliably.",
            degraded=True, latency_ms=(time.time() - t0) * 1000,
        )
    ret_5d = (prices[-1] - prices[-5]) / prices[-5] * 100
    if ret_5d > 3:
        label, conf = "BULLISH", min(0.55 + abs(ret_5d) / 20, 0.95)
    elif ret_5d < -3:
        label, conf = "BEARISH", min(0.55 + abs(ret_5d) / 20, 0.95)
    else:
        label, conf = "NEUTRAL", 0.5
    reasoning = f"5-day return of {ret_5d:.2f}% classified as {label} momentum."
    return AgentOutput("MomentumAgent", "price_momentum", label, round(conf, 2),
                        reasoning, latency_ms=(time.time() - t0) * 1000)


# ---------------------------------------------------------------------------
# Agent 2: Volume anomaly
# ---------------------------------------------------------------------------
def volume_agent(market_data: dict) -> AgentOutput:
    t0 = time.time()
    vols = market_data.get("volumes")
    if not vols or len(vols) < 5:
        return AgentOutput(
            "VolumeAgent", "volume_anomaly", "INSUFFICIENT_DATA", 0.0,
            "Volume feed unavailable - cannot assess anomaly.",
            degraded=True, latency_ms=(time.time() - t0) * 1000,
        )
    hist = vols[:-1]
    today = vols[-1]
    stdev = statistics.stdev(hist) or 1
    z = (today - statistics.mean(hist)) / stdev
    if z > 1.5:
        label, conf = "VOLUME_SPIKE", min(0.5 + z / 6, 0.95)
    elif z < -1.5:
        label, conf = "VOLUME_DROUGHT", min(0.5 + abs(z) / 6, 0.95)
    else:
        label, conf = "NORMAL", 0.55
    reasoning = f"Today's volume is {z:.2f} std-dev from 10-day mean ({label})."
    return AgentOutput("VolumeAgent", "volume_anomaly", label, round(conf, 2),
                        reasoning, latency_ms=(time.time() - t0) * 1000)


# ---------------------------------------------------------------------------
# Agent 3: Sentiment
# ---------------------------------------------------------------------------
def sentiment_agent(news: list) -> AgentOutput:
    t0 = time.time()
    if not news:
        return AgentOutput(
            "SentimentAgent", "sentiment", "INSUFFICIENT_DATA", 0.0,
            "No news/social snippets available for this ticker in the window.",
            degraded=True, latency_ms=(time.time() - t0) * 1000,
        )
    pos_words = {"beats", "surge", "growth", "upgrade", "strong", "record", "profit"}
    neg_words = {"miss", "fraud", "downgrade", "weak", "probe", "loss", "decline", "fall"}
    score = 0
    for headline in news:
        low = headline.lower()
        score += sum(w in low for w in pos_words) - sum(w in low for w in neg_words)
    if score > 0:
        label, conf = "POSITIVE", min(0.5 + score * 0.1, 0.9)
    elif score < 0:
        label, conf = "NEGATIVE", min(0.5 + abs(score) * 0.1, 0.9)
    else:
        label, conf = "NEUTRAL", 0.5
    reasoning = f"Keyword-scored {len(news)} headlines, net sentiment score {score} -> {label}."
    return AgentOutput("SentimentAgent", "sentiment", label, round(conf, 2),
                        reasoning, latency_ms=(time.time() - t0) * 1000)


# ---------------------------------------------------------------------------
# Agent 4: RAG over regulatory/earnings corpus
# ---------------------------------------------------------------------------
class RAGIndex:
    def __init__(self, corpus: List[Dict]):
        self.corpus = corpus
        self.texts = [c["text"] for c in corpus]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.texts)

    def query(self, q: str, k=2):
        qv = self.vectorizer.transform([q])
        sims = cosine_similarity(qv, self.matrix)[0]
        idx = np.argsort(sims)[::-1][:k]
        return [(self.corpus[i], float(sims[i])) for i in idx if sims[i] > 0.05]


def rag_agent(index: "RAGIndex", ticker: str) -> AgentOutput:
    t0 = time.time()
    hits = index.query(f"{ticker} regulatory filing risk earnings")
    if not hits:
        return AgentOutput(
            "RAGAgent", "filing_grounding", "NO_MATCH", 0.0,
            "No relevant filing/document found in corpus for this query.",
            degraded=True, latency_ms=(time.time() - t0) * 1000,
        )
    citations = [f"{h[0]['doc_id']}::{h[0]['section']}" for h in hits]
    summary = " | ".join(h[0]["text"][:140] for h in hits)
    conf = min(hits[0][1] + 0.3, 0.95)
    return AgentOutput(
        "RAGAgent", "filing_grounding", "GROUNDED", round(conf, 2),
        f"Retrieved {len(hits)} relevant filing excerpt(s): {summary}",
        citations=citations, latency_ms=(time.time() - t0) * 1000,
    )


# ---------------------------------------------------------------------------
# Risk profiling (personalization)
# ---------------------------------------------------------------------------
RISK_PROFILES = {
    "conservative": {"momentum_weight": 0.6, "sentiment_weight": 0.4, "volume_weight": 0.3,
                      "action_threshold": 0.75, "max_position_pct": 5},
    "moderate":     {"momentum_weight": 1.0, "sentiment_weight": 0.8, "volume_weight": 0.6,
                      "action_threshold": 0.60, "max_position_pct": 10},
    "aggressive":   {"momentum_weight": 1.3, "sentiment_weight": 1.1, "volume_weight": 1.0,
                      "action_threshold": 0.50, "max_position_pct": 20},
}


# ---------------------------------------------------------------------------
# Synthesis layer
# ---------------------------------------------------------------------------
def synthesize(outputs: List[AgentOutput], profile_name: str) -> dict:
    profile = RISK_PROFILES[profile_name]
    active = [o for o in outputs if not o.degraded]
    degraded = [o for o in outputs if o.degraded]

    weight_map = {
        "price_momentum": profile["momentum_weight"],
        "sentiment": profile["sentiment_weight"],
        "volume_anomaly": profile["volume_weight"],
    }

    bull_score = bear_score = 0.0
    for o in active:
        w = weight_map.get(o.dimension, 0.5)
        if o.label in ("BULLISH", "POSITIVE", "VOLUME_SPIKE"):
            bull_score += o.confidence * w
        elif o.label in ("BEARISH", "NEGATIVE"):
            bear_score += o.confidence * w

    conflict = bull_score > 0 and bear_score > 0 and abs(bull_score - bear_score) < 0.15
    net = bull_score - bear_score
    note = (f"{len(degraded)} agent(s) degraded ({', '.join(d.agent for d in degraded)}); "
            f"recommendation confidence reduced.") if degraded else ""

    if conflict:
        action = "HOLD / WATCH"
        reasoning = (f"Conflicting signals detected (bull={bull_score:.2f} vs "
                     f"bear={bear_score:.2f}). {note} Recommend no new position "
                     f"until signals align.")
        conf = 0.4
    elif net >= profile["action_threshold"]:
        action = f"CONSIDER BUY (max {profile['max_position_pct']}% of portfolio)"
        reasoning = (f"Weighted bullish score {net:.2f} clears {profile_name} "
                     f"threshold {profile['action_threshold']}. {note}")
        conf = min(net, 0.95)
    elif net <= -profile["action_threshold"]:
        action = "CONSIDER REDUCE/AVOID"
        reasoning = (f"Weighted bearish score {abs(net):.2f} clears {profile_name} "
                     f"threshold. {note}")
        conf = min(abs(net), 0.95)
    else:
        action = "HOLD"
        reasoning = f"Net weighted score {net:.2f} below action threshold for {profile_name} profile. {note}"
        conf = 0.5

    citations = []
    for o in outputs:
        citations.extend(o.citations)

    return {
        "action": action,
        "confidence": round(conf, 2),
        "reasoning": reasoning,
        "citations": citations,
        "conflict": conflict,
        "degraded_agents": [d.agent for d in degraded],
        "profile": profile_name,
    }


# ---------------------------------------------------------------------------
# Orchestrator - dispatches agents in parallel
# ---------------------------------------------------------------------------
def run_pipeline(ticker, market_data, news, rag_index, profile_name, simulate_degraded=None):
    tasks = {
        "momentum": lambda: momentum_agent(market_data if simulate_degraded != "momentum" else {}),
        "volume": lambda: volume_agent(market_data if simulate_degraded != "volume" else {}),
        "sentiment": lambda: sentiment_agent(news if simulate_degraded != "sentiment" else []),
        "rag": lambda: rag_agent(rag_index, ticker),
    }
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {k: ex.submit(v) for k, v in tasks.items()}
        results = {k: f.result() for k, f in futures.items()}

    outputs = list(results.values())
    synthesis = synthesize(outputs, profile_name)
    return outputs, synthesis


# ---------------------------------------------------------------------------
# Performance logging
# ---------------------------------------------------------------------------
LOG_PATH = os.path.join(os.path.dirname(__file__), "session_log.csv")


def log_session(ticker, profile_name, outputs, synthesis, forward_return_mock, portfolio):
    weights = np.array(list(portfolio.values()), dtype=float)
    weights = weights / weights.sum() if weights.sum() > 0 else weights
    hhi = float(np.sum(weights ** 2))  # portfolio concentration score (0=diverse, 1=concentrated)

    avg_latency = float(np.mean([o.latency_ms for o in outputs]))
    accuracy_proxy = 1 if (
        (synthesis["action"].startswith("CONSIDER BUY") and forward_return_mock > 0)
        or (synthesis["action"].startswith("CONSIDER REDUCE") and forward_return_mock < 0)
    ) else 0

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "ticker": ticker,
        "profile": profile_name,
        "action": synthesis["action"],
        "confidence": synthesis["confidence"],
        "avg_agent_latency_ms": round(avg_latency, 2),
        "portfolio_concentration_hhi": round(hhi, 3),
        "mock_30d_forward_return_pct": round(forward_return_mock, 2),
        "directional_accuracy_proxy": accuracy_proxy,
        "degraded_agents": ";".join(synthesis["degraded_agents"]),
    }
    file_exists = os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    return row