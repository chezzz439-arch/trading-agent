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
    target_kind: str = "constructed"   # "structure" | "constructed" (hybrid)
    path_clear: bool = True            # no swing level between entry and target (bonus)


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
        path_veto: bool = True,
        hybrid: bool = False,
    ) -> None:
        if rr_ratio <= 0 or atr_multiplier <= 0:
            raise ValueError("rr_ratio and atr_multiplier must be positive")
        self.rr_ratio = rr_ratio
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.swing_lookback = swing_lookback
        # When False, accept constructed 5:1 targets even if a swing level sits
        # in the path (looser; more trades, no structural confirmation).
        self.path_veto = path_veto
        # Hybrid target (Phase 1.1): aim at the nearest swing structure when it
        # yields >= rr_ratio, else fall back to the constructed ATR target.
        # Structure is then a confirmation bonus (path_clear), never a veto, so
        # every valid signal gets a valid target.
        self.hybrid = hybrid

    def evaluate(self, signal: Signal, df: pd.DataFrame,
                 stop_widen: float = 1.0) -> Optional[TradePlan]:
        """Build a TradePlan. ``stop_widen`` (>=1) widens the ATR stop distance —
        e.g. 1.5 in a high-VIX tape — keeping RR intact (sizing adjusts)."""
        try:
            if df is None or len(df) < self.atr_period + 1:
                return None

            atr_val = float(atr(df, self.atr_period).iloc[-1])
            if not np.isfinite(atr_val) or atr_val <= 0:
                return None

            entry = signal.entry_price
            risk_distance = self.atr_multiplier * atr_val * max(1.0, stop_widen)
            if signal.side not in ("long", "short"):
                return None
            stop = entry - risk_distance if signal.side == "long" else entry + risk_distance

            if self.hybrid:
                target, target_kind, path_clear = self._hybrid_target(
                    signal.side, df, entry, risk_distance)
            else:
                # --- Constructed target at the configured reward:risk ---- #
                target = (entry + self.rr_ratio * risk_distance if signal.side == "long"
                          else entry - self.rr_ratio * risk_distance)
                target_kind, path_clear = "constructed", True

            entry, stop, target = _round(entry), _round(stop), _round(target)
            risk_per_share = abs(entry - stop)
            # Reject non-finite levels too: a NaN/inf entry slips past the ``<= 0``
            # checks (``NaN <= 0`` is False) and would yield a NaN/inf TradePlan
            # that pollutes scores and dashboards downstream.
            if (not np.isfinite([entry, stop, target, risk_per_share]).all()
                    or risk_per_share <= 0 or stop <= 0 or target <= 0):
                return None

            # --- Legacy hard veto (only when NOT hybrid) ----------------- #
            # Hybrid never vetoes — structure is folded into the target and the
            # path_clear flag instead, so every valid signal yields a plan.
            if not self.hybrid and self.path_veto and \
                    not self._path_is_clear(signal.side, df, entry, target):
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
                target_kind=target_kind,
                path_clear=path_clear,
            )
        except Exception:
            logger.exception("RRFilter.evaluate failed for %s", getattr(signal, "symbol", "?"))
            return None

    def _hybrid_target(
        self, side: str, df: pd.DataFrame, entry: float, risk_distance: float
    ) -> tuple[float, str, bool]:
        """Pick a target: nearest swing structure if it yields >= rr_ratio, else
        the constructed ATR target. Returns (target, kind, path_clear).

        ``path_clear`` is True when no swing level sits strictly between entry and
        the chosen target — a confirmation bonus the scorer can reward, not a veto.
        """
        constructed = (entry + self.rr_ratio * risk_distance if side == "long"
                       else entry - self.rr_ratio * risk_distance)
        structure = df.iloc[:-1] if len(df) > 1 else df
        window = structure.tail(self.swing_lookback)
        if window.empty or risk_distance <= 0:
            return constructed, "constructed", True

        if side == "long":
            level = float(window["high"].max())          # nearest overhead resistance
            reaches = level > entry and (level - entry) / risk_distance >= self.rr_ratio
            if reaches:
                # structure far enough to be the target; path clear by construction
                return level, "structure", True
            # fall back beyond the (too-close) resistance; note the obstacle
            path_clear = not (entry < level < constructed)
            return constructed, "constructed", path_clear
        else:  # short
            level = float(window["low"].min())           # nearest support below
            reaches = level < entry and (entry - level) / risk_distance >= self.rr_ratio
            if reaches:
                return level, "structure", True
            path_clear = not (constructed < level < entry)
            return constructed, "constructed", path_clear

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
