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
WATCHLIST: list[str] = ["AAPL", "TSLA", "SPY", "NVDA", "BTC/USD", "ETH/USD"]
# Symbols containing "/" are routed to the crypto data/execution paths.
CRYPTO_SYMBOLS: set[str] = {"BTC/USD", "ETH/USD"}

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
RR_RATIO: float = 5.0              # minimum acceptable reward:risk
ATR_PERIOD: int = 14
ATR_MULTIPLIER: float = 1.5        # stop distance = ATR * multiplier
SWING_LOOKBACK: int = 20           # bars scanned for swing high/low target

# --------------------------------------------------------------------------- #
# Per-trade risk / position sizing
# --------------------------------------------------------------------------- #
RISK_PER_TRADE: float = 0.01       # risk 1% of equity per trade
MAX_POSITION_PCT: float = 0.10     # cap a single position at 10% of equity

# --------------------------------------------------------------------------- #
# Portfolio risk
# --------------------------------------------------------------------------- #
MAX_CONCURRENT_POSITIONS: int = 3
DAILY_LOSS_LIMIT: float = 0.03     # kill switch at -3% from day-start equity
MAX_CORRELATION: float = 0.80      # block new position too correlated to held

# --------------------------------------------------------------------------- #
# Timeframes
# --------------------------------------------------------------------------- #
SIGNAL_TIMEFRAME: str = "1Hour"            # timeframe entries are taken on
MTF_TIMEFRAMES: tuple[str, str] = ("1Hour", "4Hour")  # must agree on direction
LOOKBACK_BARS: int = 300                   # bars pulled per request

# --------------------------------------------------------------------------- #
# Run loop / logging
# --------------------------------------------------------------------------- #
SCAN_INTERVAL: int = 300           # seconds between scans (5 minutes)
LOG_DIR: str = "logs"
