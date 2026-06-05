"""Reward-to-risk filter: constructed target + swing-structure path veto.

The stop is placed ``ATR(14) * multiplier`` away from entry, and the target is
*constructed* at the configured reward:risk multiple of that stop distance
(``entry +/- rr_ratio * stop_distance``). This guarantees every accepted trade
carries the intended reward:risk (default 5:1).

Swing structure is then used as a **secondary confirmation / veto**: the trade
is only taken if the path from entry to the constructed target is clear — i.e.
no major swing high (for longs) or swing low (for shorts) sits *between* entry
and the target where it would likely stall the move. A breakout into clear
("blue sky") space, or a target that sits below the nearest major resistance,
passes; a target with a prior swing level parked in the middle of its path is
rejected.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.signals.strategy import Signal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TradePlan:
    symbol: str
    side: str             # "long" or "short"
    entry: float
    stop: float
    target: float
    rr: float
    risk_per_share: float
    atr: float
    reason: str


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range using Wilder's smoothing."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False).mean()


def swing_high(df: pd.DataFrame, lookback: int) -> float:
    """Highest high over the last ``lookback`` bars."""
    return float(df["high"].tail(lookback).max())


def swing_low(df: pd.DataFrame, lookback: int) -> float:
    """Lowest low over the last ``lookback`` bars."""
    return float(df["low"].tail(lookback).min())


def _round(value: float) -> float:
    # Alpaca rejects equity prices with > 2 decimal places (>= $1).
    return round(value, 2)


class RRFilter:
    def __init__(
        self,
        rr_ratio: float = 5.0,
        atr_period: int = 14,
        atr_multiplier: float = 1.5,
        swing_lookback: int = 100,
    ) -> None:
        if rr_ratio <= 0 or atr_multiplier <= 0:
            raise ValueError("rr_ratio and atr_multiplier must be positive")
        self.rr_ratio = rr_ratio
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.swing_lookback = swing_lookback

    def evaluate(self, signal: Signal, df: pd.DataFrame) -> Optional[TradePlan]:
        """Build a 5:1 TradePlan if the path to target is structurally clear."""
        try:
            if df is None or len(df) < self.atr_period + 1:
                return None

            atr_val = float(atr(df, self.atr_period).iloc[-1])
            if not np.isfinite(atr_val) or atr_val <= 0:
                return None

            entry = signal.entry_price
            risk_distance = self.atr_multiplier * atr_val

            # --- Constructed target at the configured reward:risk -------- #
            if signal.side == "long":
                stop = entry - risk_distance
                target = entry + self.rr_ratio * risk_distance
            elif signal.side == "short":
                stop = entry + risk_distance
                target = entry - self.rr_ratio * risk_distance
            else:
                return None

            entry, stop, target = _round(entry), _round(stop), _round(target)
            risk_per_share = abs(entry - stop)
            if risk_per_share <= 0 or stop <= 0 or target <= 0:
                return None

            # --- Secondary confirmation: structural path must be clear --- #
            if not self._path_is_clear(signal.side, df, entry, target):
                logger.info(
                    "%s: rejected, structural resistance/support blocks path "
                    "to target (entry=%.2f target=%.2f)",
                    signal.symbol, entry, target,
                )
                return None

            rr = abs(target - entry) / risk_per_share
            return TradePlan(
                symbol=signal.symbol,
                side=signal.side,
                entry=entry,
                stop=stop,
                target=target,
                rr=round(rr, 2),
                risk_per_share=round(risk_per_share, 4),
                atr=round(atr_val, 4),
                reason=signal.reason,
            )
        except Exception:
            logger.exception("RRFilter.evaluate failed for %s", getattr(signal, "symbol", "?"))
            return None

    def _path_is_clear(
        self, side: str, df: pd.DataFrame, entry: float, target: float
    ) -> bool:
        """True if no major swing level sits between entry and the target.

        Structure is measured *before* the current (signal) bar so the breakout
        bar's own extreme doesn't count as resistance against itself.

        * Long  — blocked if the highest prior swing high lies strictly between
          entry and target (overhead resistance parked in the path). A breakout
          to new highs (resistance below entry) or a target below the nearest
          major resistance both pass.
        * Short — the mirror image using the lowest prior swing low as support.
        """
        structure = df.iloc[:-1] if len(df) > 1 else df
        window = structure.tail(self.swing_lookback)
        if window.empty:
            return True  # no structure to contradict the trade

        if side == "long":
            resistance = float(window["high"].max())
            blocked = entry < resistance < target
        else:  # short
            support = float(window["low"].min())
            blocked = target < support < entry
        return not blocked
