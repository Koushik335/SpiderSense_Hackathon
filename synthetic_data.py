"""
Synthetic data generators.
The PS explicitly allows 'simulated market data' and 'equivalent synthetic documents',
so we generate deterministic-but-varied fake data instead of hitting real APIs.
This removes all internet/rate-limit risk during your live demo.
"""
import random

TICKERS = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ZOMATO"]


def generate_market_data(ticker, seed=None, crash=False):
    """30-day synthetic price + volume series for a ticker."""
    rnd = random.Random(seed or ticker)
    base = rnd.uniform(500, 3000)
    prices = [base]
    for _ in range(29):
        drift = -0.03 if crash else rnd.uniform(-0.02, 0.025)
        prices.append(prices[-1] * (1 + drift))
    volumes = [int(rnd.uniform(1e6, 5e6)) for _ in range(29)]
    volumes.append(int(volumes[-1] * (3 if crash else rnd.uniform(0.7, 1.4))))
    return {"ticker": ticker, "prices": prices, "volumes": volumes}


def generate_news(ticker, sentiment="mixed"):
    pos = [
        f"{ticker} beats Q2 earnings estimates, margins expand",
        f"Analysts upgrade {ticker} citing strong order book growth",
        f"{ticker} reports record quarterly profit",
    ]
    neg = [
        f"{ticker} shares fall after auditor flags concerns",
        f"Brokerage downgrades {ticker} on weak guidance",
        f"{ticker} faces regulatory probe over disclosure lapse",
    ]
    if sentiment == "positive":
        return pos[:2]
    if sentiment == "negative":
        return neg[:2]
    if sentiment == "none":
        return []
    return [pos[0], neg[0]]


def generate_filing_corpus():
    """Synthetic SEBI-filing-style / earnings-transcript-style documents for RAG."""
    docs = [
        ("SEBI_FILING_001", "Risk Factors",
         "RELIANCE has disclosed capex plans of Rs 75,000 crore for FY26 in its new "
         "energy and retail verticals, financed partly via internal accruals and "
         "partly via debt, marginally increasing leverage ratios."),
        ("SEBI_FILING_002", "Related Party",
         "TCS's board approved a related-party transaction with a subsidiary for "
         "cloud infrastructure services valued at Rs 1,200 crore, subject to audit "
         "committee review."),
        ("EARNINGS_TRANSCRIPT_003", "Management Commentary",
         "INFY management indicated continued softness in discretionary technology "
         "spending from BFSI clients in North America, but reiterated full-year "
         "revenue guidance."),
        ("SEBI_FILING_004", "Contingent Liability",
         "HDFCBANK disclosed a contingent liability related to a pending tax dispute "
         "of Rs 450 crore, which management believes is unlikely to materially "
         "impact financials."),
        ("EARNINGS_TRANSCRIPT_005", "Growth Outlook",
         "ZOMATO's management flagged rising cash burn in the quick-commerce "
         "vertical (Blinkit) even as order volumes grew, citing an aggressive "
         "dark-store expansion strategy."),
    ]
    return [{"doc_id": d, "section": s, "text": t} for d, s, t in docs]


def generate_portfolio():
    return {"RELIANCE": 40000, "TCS": 25000, "INFY": 15000, "HDFCBANK": 10000, "ZOMATO": 10000}
