"""Realistic sample data for the beginner-friendly dashboard.

When the live account is flat (no positions / no trade history yet), the
dashboard falls back to this sample dataset so every page looks full and real
for demonstration. It is always clearly labelled as sample data in the UI.

All values are static literals (no randomness/time calls) so the demo is
deterministic and the pages render identically every load.
"""

from __future__ import annotations


def _equity_curve() -> list[dict]:
    # ~90 sessions drifting from 100,000 up to ~100,234 with realistic wiggles.
    pts = [
        100000, 100120, 99980, 100210, 100340, 100180, 100090, 100300, 100450,
        100380, 100260, 100410, 100590, 100470, 100330, 100520, 100680, 100540,
        100390, 100610, 100480, 100350, 100270, 100160, 100040, 99910, 99830,
        99950, 100080, 100240, 100190, 100070, 100210, 100360, 100500, 100430,
        100310, 100180, 100330, 100470, 100600, 100520, 100390, 100250, 100130,
        100040, 99960, 100110, 100270, 100190, 100340, 100480, 100410, 100280,
        100150, 100300, 100450, 100370, 100240, 100110, 100260, 100400, 100330,
        100210, 100090, 100230, 100380, 100300, 100170, 100050, 100190, 100340,
        100270, 100150, 100290, 100430, 100360, 100240, 100130, 100280, 100410,
        100340, 100220, 100110, 100250, 100390, 100310, 100190, 100234.5,
    ]
    return [{"t": i, "equity": v} for i, v in enumerate(pts)]


def build_sample() -> dict:
    return {
        "is_sample": True,
        "started": 100000.0,
        "equity": 100234.50,
        "daily_pnl": 234.50,
        "daily_pct": 0.23,
        "all_time_pnl": 234.50,
        "all_time_pct": 0.23,
        "win_rate": 0.67,
        "equity_history": _equity_curve(),
        "positions": [
            {
                "emoji": "🍎", "name": "Apple", "symbol": "AAPL", "shares": 32,
                "paid": 182.45, "now": 187.20, "cost": 5838.40, "value": 5990.40,
                "pnl": 152.00, "pnl_pct": 2.6, "target": 198.70, "stop": 179.20,
                "progress": 0.65, "score": 78,
                "reasons": ["Strong upward trend", "Gaining momentum",
                            "High trading volume confirming move"],
            },
            {
                "emoji": "🎮", "name": "Nvidia", "symbol": "NVDA", "shares": 18,
                "paid": 210.10, "now": 214.75, "cost": 3781.80, "value": 3865.50,
                "pnl": 83.70, "pnl_pct": 2.2, "target": 233.40, "stop": 204.90,
                "progress": 0.33, "score": 81,
                "reasons": ["Breaking to new highs", "Sector leading the market",
                            "Volume 2x the average"],
            },
        ],
        "trades": [
            {"emoji": "🍎", "name": "Apple", "symbol": "AAPL", "win": True,
             "bought_on": "Jun 1", "bought_at": 182.45, "sold_on": "Jun 4",
             "sold_at": 191.20, "days": 3, "pnl": 280.00, "pnl_pct": 4.8,
             "score": 78, "note": ""},
            {"emoji": "🟠", "name": "Bitcoin", "symbol": "BTC/USD", "win": False,
             "bought_on": "Jun 2", "bought_at": 43200.0, "sold_on": "Jun 3",
             "sold_at": 42850.0, "days": 1, "pnl": -120.00, "pnl_pct": -0.8,
             "score": 72, "note": "Market reversed direction unexpectedly"},
            {"emoji": "🪟", "name": "Microsoft", "symbol": "MSFT", "win": True,
             "bought_on": "May 26", "bought_at": 418.00, "sold_on": "May 30",
             "sold_at": 436.40, "days": 4, "pnl": 221.00, "pnl_pct": 4.4,
             "score": 75, "note": ""},
            {"emoji": "🛒", "name": "Amazon", "symbol": "AMZN", "win": True,
             "bought_on": "May 20", "bought_at": 244.00, "sold_on": "May 27",
             "sold_at": 255.30, "days": 7, "pnl": 198.00, "pnl_pct": 4.6,
             "score": 73, "note": ""},
            {"emoji": "💳", "name": "Visa", "symbol": "V", "win": False,
             "bought_on": "May 18", "bought_at": 291.00, "sold_on": "May 19",
             "sold_at": 286.60, "days": 1, "pnl": -125.00, "pnl_pct": -1.5,
             "score": 71, "note": "Hit its safety net when the market dipped"},
            {"emoji": "⚡", "name": "Tesla", "symbol": "TSLA", "win": True,
             "bought_on": "May 12", "bought_at": 334.00, "sold_on": "May 16",
             "sold_at": 351.20, "days": 4, "pnl": 207.00, "pnl_pct": 5.1,
             "score": 76, "note": ""},
        ],
        "watching": [
            {"emoji": "🍎", "name": "Apple", "symbol": "AAPL", "price": 187.20,
             "chg": 1.2, "view": "BULLISH", "score": 78, "status": "OWNED",
             "reasons": ["Trending upward", "Strong momentum", "Good trading volume"]},
            {"emoji": "🎮", "name": "Nvidia", "symbol": "NVDA", "price": 214.75,
             "chg": 1.8, "view": "BULLISH", "score": 81, "status": "OWNED",
             "reasons": ["New all-time highs", "Leading its sector", "Heavy volume"]},
            {"emoji": "⚡", "name": "Tesla", "symbol": "TSLA", "price": 423.70,
             "chg": 0.9, "view": "BULLISH", "score": 71, "status": "WATCHING",
             "reasons": ["Trend turning up", "Momentum building"]},
            {"emoji": "🪟", "name": "Microsoft", "symbol": "MSFT", "price": 427.34,
             "chg": 0.4, "view": "NEUTRAL", "score": 58, "status": "WATCHING",
             "reasons": ["Drifting sideways", "Waiting for a clearer move"]},
            {"emoji": "🛒", "name": "Amazon", "symbol": "AMZN", "price": 250.02,
             "chg": -0.3, "view": "NEUTRAL", "score": 55, "status": "WATCHING",
             "reasons": ["No clear direction", "Needs a stronger signal"]},
            {"emoji": "🔍", "name": "Google", "symbol": "GOOGL", "price": 358.99,
             "chg": 0.6, "view": "BULLISH", "score": 64, "status": "WATCHING",
             "reasons": ["Slow steady uptrend", "Volume a bit light"]},
            {"emoji": "👥", "name": "Meta", "symbol": "META", "price": 612.40,
             "chg": -1.1, "view": "BEARISH", "score": 38, "status": "AVOIDING",
             "reasons": ["Trending down", "Losing momentum"]},
            {"emoji": "💻", "name": "AMD", "symbol": "AMD", "price": 142.20,
             "chg": -0.7, "view": "BEARISH", "score": 42, "status": "AVOIDING",
             "reasons": ["Below key averages", "Sector weak"]},
            {"emoji": "📺", "name": "Netflix", "symbol": "NFLX", "price": 815.20,
             "chg": 0.8, "view": "NEUTRAL", "score": 57, "status": "WATCHING",
             "reasons": ["Choppy range", "Waiting for breakout"]},
            {"emoji": "🏦", "name": "JPMorgan", "symbol": "JPM", "price": 289.10,
             "chg": 0.3, "view": "NEUTRAL", "score": 52, "status": "WATCHING",
             "reasons": ["Sideways", "No edge yet"]},
            {"emoji": "🟠", "name": "Bitcoin", "symbol": "BTC/USD", "price": 67234.0,
             "chg": -0.8, "view": "NEUTRAL", "score": 54, "status": "WATCHING",
             "reasons": ["No clear direction", "Needs stronger signal"]},
            {"emoji": "💎", "name": "Ethereum", "symbol": "ETH/USD", "price": 3420.0,
             "chg": -1.4, "view": "BEARISH", "score": 41, "status": "AVOIDING",
             "reasons": ["Pulling back", "Weak vs Bitcoin"]},
            {"emoji": "☀️", "name": "Solana", "symbol": "SOL/USD", "price": 178.30,
             "chg": 2.1, "view": "BULLISH", "score": 66, "status": "WATCHING",
             "reasons": ["Strongest crypto today", "Momentum building"]},
        ],
        "activity": [
            {"icon": "🟢", "ago": "2h ago",
             "text": "Bot BOUGHT Apple · 32 shares at $182.45 · risking $150"},
            {"icon": "⚡", "ago": "1h ago",
             "text": "Apple hit +2x profit · Stop moved to breakeven · You can't lose now"},
            {"icon": "👀", "ago": "45m ago",
             "text": "Nvidia looked good (scored 84) but not enough room to profit · watching"},
            {"icon": "🔍", "ago": "5m ago",
             "text": "Scanned all markets · nothing new · waiting for the perfect setup"},
        ],
        "bot": {
            "running": True, "last_scan_min": 2, "next_scan_min": 3,
            "uptime": "4 hours 32 minutes", "scans_today": 54,
            "setups_looked": 702, "trades_taken": 1, "setups_rejected": 701,
            "daily_loss_used": 0.0, "daily_loss_limit": 3000.0,
            "telegram": True, "broker": True, "logging": True,
            "min_score": 70, "rr": 4.0, "risk_pct": 1.0,
        },
        "performance": {
            "sharpe": 1.33, "max_drawdown_pct": -3.73, "max_drawdown_dollar": 3730,
            "avg_winner": 280.0, "avg_loser": -120.0,
            "monthly": [  # (day-of-month label, pnl) for a sample month
                ("2", 45), ("3", 120), ("4", -30), ("5", 89), ("6", 0),
                ("9", 60), ("10", -45), ("11", 110), ("12", 35), ("13", 70),
                ("16", -20), ("17", 95), ("18", -125), ("19", 40), ("20", 198),
                ("23", 55), ("24", -30), ("25", 80), ("26", 221), ("27", 15),
                ("30", 50), ("31", 234),
            ],
        },
    }
