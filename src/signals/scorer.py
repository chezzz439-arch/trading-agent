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

# Pure-momentum-long-only weights. Mean-reversion was removed from `statistical`
# (15->12) and `regime` (15->8); the freed 10 points fund the new
# `relative_strength` dimension (must outperform SPY). Sums to 100.
MAX_POINTS = {
    "technical": 20, "momentum": 15, "mtf": 15, "statistical": 12,
    "regime": 8, "ml": 10, "risk_reward": 10, "relative_strength": 10,
}


@dataclass
class TradeScore:
    symbol: str
    side: str
    total: float = 0.0
    breakdown: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    min_score: float = 70.0
    # Hard momentum gates: a short (no shorts ever) or a long not outperforming
    # SPY is vetoed regardless of score. Both entry paths check `passed`, so the
    # gate applies live and in backtest from this one place.
    rs_veto: bool = False

    @property
    def passed(self) -> bool:
        return self.total >= self.min_score and not self.rs_veto


class MasterScorer:
    def __init__(self, min_score: float = 70.0, rr_target: float = 5.0):
        self.min_score = min_score
        # Reward:risk that earns full marks — keep in sync with the RR filter's
        # ratio so a target-meeting trade isn't penalised when the target moves.
        self.rr_target = rr_target

    def prerank_score(self, symbol: str, side: str, technical) -> float:
        """Cheap technical+momentum-only score (0–35) for fast candidate ranking.

        Uses no network/heavy inputs, so it can rank the whole universe quickly;
        the full pipeline then runs on the top-ranked names.
        """
        s = TradeScore(symbol=symbol, side=side, min_score=self.min_score)
        return round(self._technical(side, technical, s) + self._momentum(side, technical, s), 1)

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
        research: object = None,
    ) -> TradeScore:
        s = TradeScore(symbol=symbol, side=side, min_score=self.min_score)
        b = s.breakdown
        # Hard gate 1: long-only, no shorts ever.
        if side != "long":
            s.rs_veto = True
            s.notes.append("vetoed: shorts disabled (long-only momentum)")
        b["technical"] = self._technical(side, technical, s)
        b["momentum"] = self._momentum(side, technical, s)
        b["mtf"] = self._mtf(side, mtf, s)
        b["statistical"] = self._statistical(side, quant, s)
        b["regime"] = self._regime(side, regime, s)
        b["ml"] = self._ml(side, ml, s)
        b["risk_reward"] = self._risk_reward(plan, s)
        b["relative_strength"] = self._relative_strength(quant, s)
        # Research is additive to the technical score and already clamped to
        # +/-25 by the ResearchEngine. It's side-aware: a bullishness score added
        # for longs, sign-flipped for shorts. Live-only (None in backtest -> 0).
        # The final total is bounded to [0, 100].
        _ap = getattr(research, "applied_points", None)
        b["research"] = float(_ap(side)) if callable(_ap) else 0.0
        s.total = round(min(100.0, max(0.0, sum(b.values()))), 1)
        return s

    # ------------------------------------------------------------------ #
    def _technical(self, side, t: Optional[TechnicalResult], s) -> float:
        if t is None:
            return MAX_POINTS["technical"] * 0.5
        pts = 0.0
        # MA alignment (max 10): price above the key EMAs (uptrend structure).
        above = t.signals.get("above_key_mas", 0)
        pts += above / 4 * 10
        # Bullish pattern confirmation (max 5).
        if t.signals.get("bullish_pattern"):
            pts += 5
        # Volume confirmation (max 5): entry-day volume above its 20-day average.
        if t.signals.get("volume_confirms"):
            pts += 5
        return round(min(pts, MAX_POINTS["technical"]), 1)

    def _momentum(self, side, t: Optional[TechnicalResult], s) -> float:
        if t is None:
            return MAX_POINTS["momentum"] * 0.5
        pts = 0.0
        # Bullish RSI without being overbought (max 6).
        if not t.signals.get("rsi_overbought"):
            pts += 6 if t.signals.get("rsi_bull") else 3
        # MACD histogram positive (max 5).
        if t.signals.get("macd_bull"):
            pts += 5
        # Stochastic has room to run up (max 4).
        k = t.values.get("stoch_k")
        if k is not None and k < 80:
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

    def _statistical(self, side, q: Optional[QuantResult], s) -> float:
        """Pure-trend statistical edge (max 12). Mean-reversion / oversold-bounce
        components removed — this strategy only rewards trend persistence."""
        if q is None:
            return MAX_POINTS["statistical"] * 0.5
        pts = 0.0
        v = q.values
        # Trending (persistent) series favoured — Hurst > 0.55 (max 5).
        hurst = v.get("hurst")
        if hurst is not None:
            if hurst > 0.55:
                pts += 5
            elif hurst >= 0.45:
                pts += 1
        # Positive regression slope = uptrend (max 4).
        slope = v.get("slope_pct") or 0
        if slope > 0:
            pts += 4
        # Positive autocorrelation = trend continuation (max 3).
        ac = v.get("autocorr_lag1")
        if ac is not None and ac > 0:
            pts += 3
        return round(min(pts, MAX_POINTS["statistical"]), 1)

    def _relative_strength(self, q: Optional[QuantResult], s) -> float:
        """Relative strength vs SPY (max 10) + hard gate. A long must outperform
        SPY on both the 20- and 60-bar horizons; if it doesn't, veto the trade.
        Neutral (half marks, no veto) when SPY is unavailable (rs is None)."""
        if q is None:
            return MAX_POINTS["relative_strength"] * 0.5
        rs20 = q.values.get("rel_strength_20")
        rs60 = q.values.get("rel_strength_60")
        outperf = q.values.get("rs_outperform")
        if outperf is None:                       # no market series -> neutral
            s.notes.append("rs: SPY unavailable -> neutral")
            return MAX_POINTS["relative_strength"] * 0.5
        if not outperf:                           # hard gate: must beat SPY
            s.rs_veto = True
            s.notes.append(f"vetoed: lagging SPY (rs20={rs20:+.3f}, rs60={rs60:+.3f})")
            return 0.0
        # Scale points by the strength of outperformance (cap at +10% excess).
        pts = 5 + min(5.0, (max(rs20, 0) + max(rs60, 0)) / 0.10 * 5)
        return round(min(pts, MAX_POINTS["relative_strength"]), 1)

    def _regime(self, side, r: Optional[Regime], s) -> float:
        """Reward trend regimes only (max 8). Mean-reversion regimes no longer
        earn points — this strategy only trades momentum."""
        if r is None:
            return MAX_POINTS["regime"] * 0.5
        if r.trend == "strong_trend" and r.strategy == "momentum":
            return float(MAX_POINTS["regime"])
        if r.trend == "weak_trend" and r.strategy == "momentum":
            return 4.0
        return 0.0

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
        t = self.rr_target
        if plan.rr >= t:
            # Full points at the target RR, small bonus capped at 10 for better.
            return round(min(MAX_POINTS["risk_reward"], 8 + (plan.rr - t) * 0.5), 1)
        return round(max(0.0, plan.rr / t * 6), 1)
