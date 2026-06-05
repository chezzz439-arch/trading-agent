"""Phase 3 — Regime detection.

Classifies the current environment along five axes (volatility, trend, market,
momentum, volume) and combines them into a single label plus a recommended
strategy family (momentum vs mean-reversion) and parameter set. The scorer and
risk modules use the regime to decide which signals are active and how much
size is appropriate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

import ta

logger = logging.getLogger(__name__)


@dataclass
class Regime:
    volatility: str = "unknown"   # low / medium / high / extreme
    trend: str = "unknown"        # strong_trend / weak_trend / ranging
    market: str = "unknown"       # risk_on / risk_off / neutral
    momentum: str = "unknown"     # accelerating / decelerating / reversing
    volume: str = "unknown"       # accumulation / distribution / neutral
    label: str = "unknown"
    strategy: str = "none"        # momentum / mean_reversion / none
    params: dict = field(default_factory=dict)

    def favors(self, side: str) -> bool:
        """Does this regime support taking a trade of strategy ``side``?"""
        return self.strategy == side


# Regime-specific parameter sets selected by the combined label.
_PARAM_SETS = {
    "momentum": {"rr_ratio": 5.0, "atr_multiplier": 1.5, "prefer": "trend_following"},
    "mean_reversion": {"rr_ratio": 3.0, "atr_multiplier": 1.0, "prefer": "fade_extremes"},
    "none": {"rr_ratio": 5.0, "atr_multiplier": 1.5, "prefer": "stand_aside"},
}


class RegimeDetector:
    def detect(
        self,
        df: pd.DataFrame,
        vix: Optional[float] = None,
        spy_df: Optional[pd.DataFrame] = None,
    ) -> Regime:
        try:
            if df is None or len(df) < 60:
                return Regime()
            r = Regime()
            h, l, c = df["high"], df["low"], df["close"]

            # --- Volatility regime: ATR percentile vs its own history ---- #
            atr = ta.volatility.AverageTrueRange(h, l, c, 14).average_true_range()
            atr_now = atr.iloc[-1]
            atr_pct = float((atr.dropna() < atr_now).mean() * 100)
            if atr_pct > 95:
                r.volatility = "extreme"
            elif atr_pct > 75:
                r.volatility = "high"
            elif atr_pct < 25:
                r.volatility = "low"
            else:
                r.volatility = "medium"

            # --- Trend regime: ADX -------------------------------------- #
            adx = ta.trend.ADXIndicator(h, l, c, 14).adx().iloc[-1]
            adx = float(adx) if np.isfinite(adx) else 0.0
            if adx > 35:
                r.trend = "strong_trend"
            elif adx >= 25:
                r.trend = "weak_trend"
            else:
                r.trend = "ranging"

            # --- Market regime: VIX level + SPY trend ------------------- #
            spy_bull = None
            if spy_df is not None and len(spy_df) >= 50:
                sc = spy_df["close"]
                spy_bull = bool(sc.iloc[-1] > sc.ewm(span=50, adjust=False).mean().iloc[-1])
            if vix is not None:
                if vix > 28 or spy_bull is False:
                    r.market = "risk_off"
                elif vix < 18 and (spy_bull is None or spy_bull):
                    r.market = "risk_on"
                else:
                    r.market = "neutral"
            elif spy_bull is not None:
                r.market = "risk_on" if spy_bull else "risk_off"

            # --- Momentum regime: slope of ROC -------------------------- #
            roc = ta.momentum.ROCIndicator(c, 12).roc()
            roc_slope = roc.iloc[-1] - roc.iloc[-5] if len(roc) > 5 else 0.0
            if roc.iloc[-1] * roc.iloc[-5] < 0:
                r.momentum = "reversing"
            elif roc_slope > 0:
                r.momentum = "accelerating"
            else:
                r.momentum = "decelerating"

            # --- Volume regime: OBV slope + CMF ------------------------- #
            obv = ta.volume.OnBalanceVolumeIndicator(c, df["volume"]).on_balance_volume()
            obv_slope = obv.iloc[-1] - obv.iloc[-10] if len(obv) > 10 else 0.0
            cmf = ta.volume.ChaikinMoneyFlowIndicator(h, l, c, df["volume"], 20).chaikin_money_flow().iloc[-1]
            if obv_slope > 0 and cmf > 0:
                r.volume = "accumulation"
            elif obv_slope < 0 and cmf < 0:
                r.volume = "distribution"
            else:
                r.volume = "neutral"

            # --- Combine ------------------------------------------------ #
            if r.trend in ("strong_trend", "weak_trend") and r.volatility != "extreme":
                r.strategy = "momentum"
            elif r.trend == "ranging" and r.volatility in ("low", "medium"):
                r.strategy = "mean_reversion"
            else:
                r.strategy = "none"
            r.params = dict(_PARAM_SETS[r.strategy])
            r.label = f"{r.volatility}_vol/{r.trend}/{r.market}/{r.strategy}"
            return r
        except Exception:
            logger.exception("RegimeDetector.detect failed")
            return Regime()
