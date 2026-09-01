# 🕷️ SPIDER-SENSE Financial Intelligence System
> **PS-01: Multi-Agent Autonomous Financial Intelligence System for Retail Investors**  
> **Team:** YOLOTECH | **Event:** HackVerse 2026, VIT Chennai

---

## 🚀 System Architecture

SPIDER-SENSE bridges the retail investment infrastructure gap through a multi-agent orchestration layer that executes parallel domain research in sub-150ms and synthesizes explainable, behavioral-tailored recommendations.

### 1. Parallel Domain Agents
* **Technical Momentum Agent:** Evaluates multi-timeframe price action, RSI thresholds, and volume anomaly Z-scores ($\sigma$).
* **Regulatory Fundamental Agent (RAG):** Performs semantic search over SEBI corporate filings, Capex disclosures, and earnings transcripts with verbatim source chunk citations.
* **Sentiment & Institutional Flow Agent:** Tracks real-time FII/DII net flows and Options Chain Put-Call Ratio (PCR) skew.

### 2. Behavioral Personalization Engine
* Dynamically re-weights agent signals based on the user's risk tier, loss tolerance boundaries, and portfolio concentration score (Herfindahl-Hirschman Index - HHI).
* **Identical Signal, Differential Output:** High-momentum assets (e.g., $RSI > 80$) trigger `AVOID / CAPITAL PRESERVATION` for conservative users, but generate `MOMENTUM BREAKOUT BUY` with strict trailing stop-loss orders for aggressive traders.

### 3. Degraded Data Fault Tolerance
* Automatically detects missing filings or corrupted feeds, drops ungrounded speculative confidence to fallback baselines ($35\%$), and alerts the investor transparently without pipeline crashes.

---

## 🛠️ Quickstart

```bash
pip install -r requirements.txt
python app.py