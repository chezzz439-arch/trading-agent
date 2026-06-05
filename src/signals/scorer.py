"""Phase 7 — Master signal scorer.

Fuses every analysis phase into a single 0–100 conviction score for a proposed
trade direction, across seven weighted dimensions:

    Technical alignment   20
    Momentum quality      15
    MTF confluence        15
    Statistical edge      15
    Regime fit            15
    ML confidence         10
    Risk/reward           10

Only trades scoring >= ``min_score`` (default 70) proceed to execution. Missing
components (e.g. ML or MTF unavailable in a backtest) score *neutral* for their
slice rather than zero, so absence of a live-only input doesn't veto a trade.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from src.signals.ml_signals import MLPrediction
from src.signals.mtf import MTFResult
from src.signals.quant import QuantResult
from src.signals.regime import Regime
from src.signals.rr_filter import TradePlan
from src.signals.technical import TechnicalResult

logger = logging.getLogger(__name__)

MAX_POINTS = {
    "technical": 20, "momentum": 15, "mtf": 15, "statistical": 15,
    "regime": 15, "ml": 10, "risk_reward": 10,
}


@dataclass
class TradeScore:
    symbol: str
    side: str
    total: float = 0.0
    breakdown: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    min_score: float = 70.0

    @property
    def passed(self) -> bool:
        return self.total >= self.min_score


class MasterScorer:
    def __init__(self, min_score: float = 70.0):
        self.min_score = min_score

    def score(
        self,
        symbol: str,
        side: str,
        *,
        technical: Optional[TechnicalResult] = None,
        quant: Optional[QuantResult] = None,
        regime: Optional[Regime] = None,
        mtf: Optional[MTFResult] = None,
        ml: Optional[MLPrediction] = None,
        plan: Optional[TradePlan] = None,
    ) -> TradeScore:
        s = TradeScore(symbol=symbol, side=side, min_score=self.min_score)
        b = s.breakdown
        b["technical"] = self._technical(side, technical, s)
        b["momentum"] = self._momentum(side, technical, s)
        b["mtf"] = self._mtf(side, mtf, s)
        b["statistical"] = self._statistical(side, quant, regime, s)
        b["regime"] = self._regime(side, regime, s)
        b["ml"] = self._ml(side, ml, s)
        b["risk_reward"] = self._risk_reward(plan, s)
        s.total = round(sum(b.values()), 1)
        return s

    # ------------------------------------------------------------------ #
    def _technical(self, side, t: Optional[TechnicalResult], s) -> float:
        if t is None:
            return MAX_POINTS["technical"] * 0.5
        pts = 0.0
        # MA alignment (max 10): how many of the key EMAs price sits on the
        # correct side of.
        above = t.signals.get("above_key_mas", 0)
        ma_pts = above / 4 * 10 if side == "long" else (4 - above) / 4 * 10
        pts += ma_pts
        # Pattern confirmation (max 5).
        if side == "long" and t.signals.get("bullish_pattern"):
            pts += 5
        elif side == "short" and t.signals.get("bearish_pattern"):
            pts += 5
        # Volume confirmation (max 5).
        if t.signals.get("volume_confirms"):
            pts += 5
        return round(min(pts, MAX_POINTS["technical"]), 1)

    def _momentum(self, side, t: Optional[TechnicalResult], s) -> float:
        if t is None:
            return MAX_POINTS["momentum"] * 0.5
        pts = 0.0
        # RSI not stretched against us (max 6).
        if side == "long" and not t.signals.get("rsi_overbought"):
            pts += 6 if t.signals.get("rsi_bull") else 3
        elif side == "short" and not t.signals.get("rsi_oversold"):
            pts += 6 if t.signals.get("rsi_bear") else 3
        # MACD histogram direction (max 5).
        if side == "long" and t.signals.get("macd_bull"):
            pts += 5
        elif side == "short" and t.signals.get("macd_bear"):
            pts += 5
        # Stochastic position (max 4): room to move in our direction.
        k = t.values.get("stoch_k")
        if k is not None:
            if side == "long" and k < 80:
                pts += 4
            elif side == "short" and k > 20:
                pts += 4
        return round(min(pts, MAX_POINTS["momentum"]), 1)

    def _mtf(self, side, m: Optional[MTFResult], s) -> float:
        if m is None or not m.views:
            s.notes.append("mtf: unavailable -> neutral")
            return MAX_POINTS["mtf"] * 0.5
        if m.dominant_direction != side:
            return 0.0
        # Scale by how many timeframes agree (out of up to 5).
        return round(min(m.confluence_score / 5 * MAX_POINTS["mtf"], MAX_POINTS["mtf"]), 1)

    def _statistical(self, side, q: Optional[QuantResult], regime: Optional[Regime], s) -> float:
        if q is None:
            return MAX_POINTS["statistical"] * 0.5
        pts = 0.0
        v = q.values
        is_mr = regime is not None and regime.strategy == "mean_reversion"
        hurst = v.get("hurst")
        # Hurst aligned with strategy type (max 6).
        if hurst is not None:
            if is_mr and hurst < 0.45:
                pts += 6
            elif not is_mr and hurst > 0.55:
                pts += 6
            elif 0.45 <= hurst <= 0.55:
                pts += 2
        # Z-score confirms (max 5): trend trade wants slope, MR wants extreme.
        z = v.get("zscore_20")
        if z is not None:
            if is_mr:
                if side == "long" and z < -1:
                    pts += 5
                elif side == "short" and z > 1:
                    pts += 5
            else:
                slope = v.get("slope_pct") or 0
                if (side == "long" and slope > 0) or (side == "short" and slope < 0):
                    pts += 5
        # Autocorrelation positive for trend continuation (max 4).
        ac = v.get("autocorr_lag1")
        if ac is not None and not is_mr and ac > 0:
            pts += 4
        elif ac is not None and is_mr and ac < 0:
            pts += 4
        return round(min(pts, MAX_POINTS["statistical"]), 1)

    def _regime(self, side, r: Optional[Regime], s) -> float:
        if r is None:
            return MAX_POINTS["regime"] * 0.5
        if r.strategy == "none":
            return 0.0
        if r.trend == "strong_trend" and r.strategy == "momentum":
            return float(MAX_POINTS["regime"])
        if r.trend == "weak_trend" and r.strategy == "momentum":
            return 10.0
        if r.strategy == "mean_reversion":
            return 11.0
        return 6.0

    def _ml(self, side, ml: Optional[MLPrediction], s) -> float:
        if ml is None or not ml.trained:
            s.notes.append("ml: unavailable -> neutral")
            return MAX_POINTS["ml"] * 0.5
        want = "up" if side == "long" else "down"
        if ml.agreement and ml.direction == want:
            return round(ml.confidence * MAX_POINTS["ml"], 1)
        if ml.agreement and ml.direction != want:
            return 0.0
        return MAX_POINTS["ml"] * 0.3  # models disagree -> low

    def _risk_reward(self, plan: Optional[TradePlan], s) -> float:
        if plan is None:
            return 0.0
        if plan.rr >= 5.0:
            # Full points at 5:1, small bonus capped at 10 for better.
            return round(min(MAX_POINTS["risk_reward"], 8 + (plan.rr - 5) * 0.5), 1)
        return round(max(0.0, plan.rr / 5 * 6), 1)
