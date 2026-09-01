"""
synthetic_data.py - Market Feeds, Regulatory Corpus & Risk Profiles
"""

MARKET_DATA = {
    "RELIANCE": {
        "price": 2980.50,
        "change_pct": 2.4,
        "rsi_14": 63.8,
        "macd": "BULLISH_CROSSOVER",
        "volume_24h": 12450000,
        "avg_volume_20d": 8100000,
        "volume_zscore": 2.85,
        "fii_flow_crores": 450.2,
        "dii_flow_crores": 180.0,
        "options_pcr": 1.15
    },
    "TATAMOTORS": {
        "price": 995.20,
        "change_pct": 3.8,
        "rsi_14": 74.2,
        "macd": "STRONG_EXPANSION",
        "volume_24h": 22100000,
        "avg_volume_20d": 11500000,
        "volume_zscore": 3.42,
        "fii_flow_crores": 680.5,
        "dii_flow_crores": 240.0,
        "options_pcr": 1.35
    },
    "INFY": {
        "price": 1780.00,
        "change_pct": -0.8,
        "rsi_14": 46.5,
        "macd": "NEUTRAL_CONSOLIDATION",
        "volume_24h": 4100000,
        "avg_volume_20d": 5200000,
        "volume_zscore": -0.65,
        "fii_flow_crores": -120.0,
        "dii_flow_crores": 95.0,
        "options_pcr": 0.88
    },
    "ZOMATO": {
        "price": 242.10,
        "change_pct": 5.1,
        "rsi_14": 81.0,
        "macd": "OVERBOUGHT_DIVERGENCE",
        "volume_24h": 45000000,
        "avg_volume_20d": 20000000,
        "volume_zscore": 4.10,
        "fii_flow_crores": 890.0,
        "dii_flow_crores": -110.0,
        "options_pcr": 1.45
    }
}

NEWS_AND_FILINGS = [
    {
        "doc_id": "SEBI-REL-2026-Q3",
        "ticker": "RELIANCE",
        "doc_type": "SEBI Q3 Earnings & Capex Disclosure",
        "text": "Reliance Industries reported 14.2% YoY growth in retail and digital EBITDA. Capex intensity in green energy transition reached INR 18,500 Cr. Promoter pledge remains 0.0%.",
        "date": "2026-02-15"
    },
    {
        "doc_id": "SEBI-TATAMOT-2026",
        "ticker": "TATAMOTORS",
        "doc_type": "SEBI Corporate Filing - Commercial Vehicles Demerger",
        "text": "Tata Motors demerger into Commercial Vehicles and Passenger Vehicles entities approved by NCLT. JLR order book stands at 148,000 units.",
        "date": "2026-02-22"
    },
    {
        "doc_id": "SEBI-INFY-2026",
        "ticker": "INFY",
        "doc_type": "SEBI Regulatory Disclosure - GenAI Pipeline",
        "text": "Infosys expanded Topaz AI engagements to 280 active enterprise clients. Large deal TCV stood at $3.2B. Attrition stabilized at 12.8%.",
        "date": "2026-02-18"
    },
    {
        "doc_id": "SEBI-ZOMATO-2026",
        "ticker": "ZOMATO",
        "doc_type": "SEBI Filings - Blinkit Quick Commerce Expansion",
        "text": "Blinkit GOV grew 118% YoY. Quick commerce dark stores increased to 950 locations. Contribution margin turned positive at +3.2%.",
        "date": "2026-02-25"
    }
]

USER_PROFILES = {
    "Conservative (Aarav - 21)": {
        "risk_tolerance": "CONSERVATIVE",
        "capital": 50000.0,
        "loss_tolerance_pct": 5.0,
        "fno_allowed": False,
        "holdings": {"RELIANCE": 0.3, "NIFTYBEES": 0.7}
    },
    "Aggressive (Vikram - 28)": {
        "risk_tolerance": "AGGRESSIVE",
        "capital": 500000.0,
        "loss_tolerance_pct": 25.0,
        "fno_allowed": True,
        "holdings": {"TATAMOTORS": 0.4, "ZOMATO": 0.3, "INFY": 0.3}
    }
}