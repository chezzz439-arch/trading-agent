"""Central configuration for the trading agent.

Every tunable parameter lives here so strategy/risk behaviour can be changed in
one place. Credentials are loaded from the environment (``.env``) via
python-dotenv; everything else is a plain module-level constant.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------- #
# Credentials / endpoints (from .env)
# --------------------------------------------------------------------------- #
ALPACA_API_KEY: str = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY: str = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL: str = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
# Anything other than the live endpoint is treated as paper trading.
PAPER: bool = "paper" in ALPACA_BASE_URL
# Free data feed for paper accounts ("iex"); "sip" needs a paid subscription.
STOCK_DATA_FEED: str = os.getenv("ALPACA_DATA_FEED", "iex")

# --------------------------------------------------------------------------- #
# Universe
# --------------------------------------------------------------------------- #
WATCHLIST: list[str] = [
    "AAPL", "TSLA", "NVDA", "SPY", "QQQ", "MSFT", "AMZN", "META", "GOOGL", "AMD",
    "BTC/USD", "ETH/USD", "SOL/USD",
]
# Symbols containing "/" are routed to the crypto data/execution paths.
CRYPTO_SYMBOLS: set[str] = {"BTC/USD", "ETH/USD", "SOL/USD"}
MARKET_PROXY: str = "SPY"          # beta/market reference

# --------------------------------------------------------------------------- #
# Strategy (EMA crossover + RSI confirmation)
# --------------------------------------------------------------------------- #
EMA_FAST: int = 20
EMA_SLOW: int = 50
RSI_PERIOD: int = 14
RSI_LONG_THRESHOLD: float = 50.0   # long requires RSI above this
RSI_SHORT_THRESHOLD: float = 50.0  # short requires RSI below this

# --------------------------------------------------------------------------- #
# Reward / risk filter
# --------------------------------------------------------------------------- #
RR_RATIO: float = 3.0              # minimum acceptable reward:risk (TEMP: lowered from 4.0 to surface live trades)
ATR_PERIOD: int = 14
ATR_MULTIPLIER: float = 1.5        # stop distance = ATR * multiplier
SWING_LOOKBACK: int = 100          # bars scanned for structural path veto
RR_PATH_VETO: bool = False         # TEMP: relaxed — path-veto was rejecting 100% of candidates, blocking all trades

# --------------------------------------------------------------------------- #
# Per-trade risk / position sizing
# --------------------------------------------------------------------------- #
RISK_PER_TRADE: float = 0.01       # risk 1% of equity per trade
MAX_POSITION_PCT: float = 0.10     # cap a single position at 10% of equity

# --------------------------------------------------------------------------- #
# Portfolio risk
# --------------------------------------------------------------------------- #
MAX_CONCURRENT_POSITIONS: int = 8
DAILY_LOSS_LIMIT: float = 0.03      # kill switch at -3% from day-start equity
WEEKLY_LOSS_LIMIT: float = 0.07     # kill switch at -7% from week-start equity
MAX_CONSECUTIVE_LOSSES: int = 5     # kill switch after N losing trades in a row
MAX_CORRELATION: float = 0.70       # block new position too correlated to held
PORTFOLIO_HEAT_MAX: float = 0.06    # max total open risk across all positions

# --------------------------------------------------------------------------- #
# Master scorer / ML
# --------------------------------------------------------------------------- #
MIN_SCORE: float = 55.0            # minimum 0-100 score required to trade
PRERANK_TOP_N: int = 20            # deep-analyze only the top-N pre-ranked names/scan
ML_ENABLED: bool = True            # XGBoost + RandomForest ensemble (LSTM TODO)
ML_RETRAIN_DAYS: int = 30          # walk-forward retrain cadence

# --------------------------------------------------------------------------- #
# Timeframes
# --------------------------------------------------------------------------- #
SIGNAL_TIMEFRAME: str = "1Hour"            # timeframe entries are taken on
MTF_TIMEFRAMES: tuple[str, ...] = ("15Min", "1Hour", "4Hour", "1Day", "1Week")
MIN_CONFLUENCE: int = 3                     # timeframes that must agree
LOOKBACK_BARS: int = 300                   # bars pulled per request

# --------------------------------------------------------------------------- #
# Run loop / logging
# --------------------------------------------------------------------------- #
SCAN_INTERVAL: int = 240           # seconds between scans (4 min — headroom for research caches warming)
LOG_DIR: str = "logs"

# --------------------------------------------------------------------------- #
# Position management (P9)
# --------------------------------------------------------------------------- #
SCALE_OUT_1_R: float = 2.0         # take first tranche at +2R
SCALE_OUT_2_R: float = 3.5         # take second tranche at +3.5R
SCALE_OUT_FRACTION: float = 0.33   # fraction of initial qty per tranche
BREAKEVEN_R: float = 2.0           # move stop to entry at +2R
TRAIL_R: float = 3.0               # start ATR-trailing the stop at +3R
TIME_EXIT_BARS: int = 10           # close a stalled position after N bars...
TIME_EXIT_MIN_R: float = 1.0       # ...if it hasn't reached this R by then

# --------------------------------------------------------------------------- #
# Universe screen (per-symbol eligibility gate before analysis)
# --------------------------------------------------------------------------- #
SCREEN_MIN_PRICE: float = 15.0        # no low-priced names
SCREEN_MIN_MARKET_CAP: float = 3e9    # large/mid-cap only
SCREEN_MIN_AVG_VOLUME: float = 1e6    # liquidity floor (shares/day)
# Absolute path so the watchlist is found regardless of the process's CWD
# (the Streamlit dashboard and scripts can launch from anywhere).
WATCHLIST_PATH: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "watchlist.json")


def load_watchlist() -> list[str]:
    """Screened watchlist from config/watchlist.json, else the static WATCHLIST."""
    import json
    try:
        with open(WATCHLIST_PATH) as f:
            syms = json.load(f).get("symbols", [])
        if syms:
            return syms
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return WATCHLIST


def load_watchlist_meta() -> dict:
    """Per-symbol metadata (name/sector/asset_class/tradable/category/size_override)."""
    import json
    try:
        with open(WATCHLIST_PATH) as f:
            return json.load(f).get("meta", {})
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


# --------------------------------------------------------------------------- #
# Options (long calls/puts on strong signals — opt-in)
# --------------------------------------------------------------------------- #
# When True, a signal scoring >= OPTIONS_MIN_SCORE buys an ATM call (long bias)
# or ATM put (short bias) instead of the stock/short. Default OFF so the live
# bot keeps trading equities until this is explicitly enabled.
OPTIONS_ENABLED: bool = False
OPTIONS_MIN_SCORE: float = 70.0      # higher conviction bar than the stock MIN_SCORE
OPTIONS_DTE_MIN: int = 30            # days-to-expiration window (inclusive)
OPTIONS_DTE_MAX: int = 45
OPTIONS_RISK_PCT: float = 0.01       # max premium spend per trade = 1% of equity
OPTIONS_MAX_POSITIONS: int = 3       # max concurrent option positions
OPTIONS_PROFIT_TARGET: float = 1.00  # take profit at +100% (premium doubles)
OPTIONS_STOP_LOSS: float = 0.50      # cut at -50% of premium paid
OPTIONS_SKIP_EARNINGS: bool = True   # never hold an option through an earnings date
OPTIONS_EXPIRY_EXIT_DAYS: int = 1    # close at <= N days to expiration (avoid worthless decay)

# --------------------------------------------------------------------------- #
# Research layer (insider / analyst / news / social + earnings)
# --------------------------------------------------------------------------- #
# Adds a clamped +/-25 research adjustment to the technical score, so strong
# research can unlock a sub-threshold trade and bad research can block one.
# Every source is error-safe (failure -> 0 points), so this can never crash the
# loop. Set False to ignore research entirely.
RESEARCH_ENABLED: bool = True

# --------------------------------------------------------------------------- #
# v2 edge-validated changes (merged from research 2026-06-07)
# --------------------------------------------------------------------------- #
# Hybrid target: aim at swing structure when it yields >= RR_RATIO, else the
# constructed ATR target; structure is a confirmation bonus, never a hard veto.
# Validated: only hybrid configs showed significant expectancy (perm-p ~0.03).
HYBRID_TARGET_ENABLED: bool = True
# Long-only: skip all short entries. Research (172 names, 4y) showed shorts were
# pure drag (-27.8R, 13% win) while longs carry the edge (+0.455R/trade, perm-p
# 0.001). Crypto is long-only on Alpaca regardless.
LONG_ONLY: bool = True

# --------------------------------------------------------------------------- #
# Monitoring
# --------------------------------------------------------------------------- #
STREAMLIT_AUTOSTART: bool = True   # launch the Streamlit dashboard from main.py
