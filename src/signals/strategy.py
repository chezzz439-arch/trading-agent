"""EMA(20/50) crossover strategy with RSI(14) confirmation.

Two entry points:

* :meth:`EMAStrategy.evaluate` — returns a :class:`Signal` only on a *fresh*
  crossover at the latest bar (used for taking trades).
* :meth:`EMAStrategy.bias`     — returns the prevailing directional bias on the
  latest bar regardless of crossover timing (used by the multi-timeframe
  alignment check).

Indicators are computed with pandas using Wilder's smoothing for RSI so results
match standard charting platforms.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Signal:
    symbol: str
    side: str            # "long" or "short"
    entry_price: float
    rsi: float
    timestamp: pd.Timestamp
    reason: str


# --------------------------------------------------------------------------- #
# Indicator helpers
# --------------------------------------------------------------------------- #
def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=period, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0)


class EMAStrategy:
    def __init__(
        self,
        fast: int = 20,
        slow: int = 50,
        rsi_period: int = 14,
        rsi_long_threshold: float = 50.0,
        rsi_short_threshold: float = 50.0,
    ) -> None:
        self.fast = fast
        self.slow = slow
        self.rsi_period = rsi_period
        self.rsi_long_threshold = rsi_long_threshold
        self.rsi_short_threshold = rsi_short_threshold

    @property
    def min_bars(self) -> int:
        return self.slow + 2

    def _indicators(self, df: pd.DataFrame):
        close = df["close"]
        return (
            ema(close, self.fast),
            ema(close, self.slow),
            rsi(close, self.rsi_period),
        )

    def evaluate(self, symbol: str, df: pd.DataFrame) -> Optional[Signal]:
        """Return a Signal on a fresh EMA crossover at the last bar, else None."""
        try:
            if df is None or len(df) < self.min_bars:
                return None
            ema_fast, ema_slow, rsi_vals = self._indicators(df)

            f_now, f_prev = ema_fast.iloc[-1], ema_fast.iloc[-2]
            s_now, s_prev = ema_slow.iloc[-1], ema_slow.iloc[-2]
            last_rsi = float(rsi_vals.iloc[-1])
            price = float(df["close"].iloc[-1])
            ts = df.index[-1]

            crossed_up = f_prev <= s_prev and f_now > s_now
            crossed_down = f_prev >= s_prev and f_now < s_now

            if crossed_up and last_rsi > self.rsi_long_threshold:
                return Signal(symbol, "long", price, last_rsi, ts,
                              f"EMA{self.fast}>EMA{self.slow} cross, RSI={last_rsi:.1f}")
            if crossed_down and last_rsi < self.rsi_short_threshold:
                return Signal(symbol, "short", price, last_rsi, ts,
                              f"EMA{self.fast}<EMA{self.slow} cross, RSI={last_rsi:.1f}")
            return None
        except Exception:
            logger.exception("Strategy.evaluate failed for %s", symbol)
            return None

    def bias(self, df: pd.DataFrame) -> Optional[str]:
        """Return prevailing direction ('long'/'short') at the last bar, or None.

        Unlike :meth:`evaluate` this does not require a crossover — it reports
        the standing relationship between the EMAs plus RSI, for trend
        alignment across timeframes.
        """
        try:
            if df is None or len(df) < self.min_bars:
                return None
            ema_fast, ema_slow, rsi_vals = self._indicators(df)
            f, s = ema_fast.iloc[-1], ema_slow.iloc[-1]
            r = float(rsi_vals.iloc[-1])
            if f > s and r > self.rsi_long_threshold:
                return "long"
            if f < s and r < self.rsi_short_threshold:
                return "short"
            return None
        except Exception:
            logger.exception("Strategy.bias failed")
            return None
