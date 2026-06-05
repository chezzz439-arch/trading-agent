"""Phase 8 — Portfolio-level risk management.

Centralises every gate and sizing rule that sits between a passing score and an
order:

* pre-trade checks (score, RR, correlation, daily-loss),
* score-scaled risk fraction (0.5% .. 2%),
* volatility- and correlation-adjusted sizing,
* portfolio heat cap (max 6% total open risk),
* dynamic stop management (breakeven after 2R, trail after 3R),
* a multi-condition kill switch (daily -3%, weekly -7%, 5 consecutive losses).

State (day/week baselines, consecutive losses) is held on the instance and
updated by the live loop. All checks are defensive and never raise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PreTradeDecision:
    allowed: bool
    risk_fraction: float = 0.0
    reasons: list = None

    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []


class PortfolioRisk:
    def __init__(
        self,
        max_positions: int = 3,
        daily_loss_limit: float = 0.03,
        weekly_loss_limit: float = 0.07,
        max_consecutive_losses: int = 5,
        max_correlation: float = 0.70,
        portfolio_heat_max: float = 0.06,
        min_score: float = 70.0,
        min_rr: float = 5.0,
    ) -> None:
        self.max_positions = max_positions
        self.daily_loss_limit = daily_loss_limit
        self.weekly_loss_limit = weekly_loss_limit
        self.max_consecutive_losses = max_consecutive_losses
        self.max_correlation = max_correlation
        self.portfolio_heat_max = portfolio_heat_max
        self.min_score = min_score
        self.min_rr = min_rr
        self._day_start_equity: Optional[float] = None
        self._week_start_equity: Optional[float] = None
        self._consecutive_losses = 0
        self.halted = False

    # ------------------------------------------------------------------ #
    # Baselines & trade outcome tracking
    # ------------------------------------------------------------------ #
    def set_day_start_equity(self, equity: float) -> None:
        self._day_start_equity = equity
        if self._week_start_equity is None:
            self._week_start_equity = equity
        logger.info("Day-start equity=$%.2f", equity)

    def set_week_start_equity(self, equity: float) -> None:
        self._week_start_equity = equity

    def record_trade_result(self, pnl: float) -> None:
        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

    # ------------------------------------------------------------------ #
    # Kill switch
    # ------------------------------------------------------------------ #
    def kill_switch_triggered(self, current_equity: float) -> bool:
        reasons = []
        if self._day_start_equity:
            dd = (self._day_start_equity - current_equity) / self._day_start_equity
            if dd >= self.daily_loss_limit:
                reasons.append(f"daily DD {dd*100:.1f}%>={self.daily_loss_limit*100:.0f}%")
        if self._week_start_equity:
            wdd = (self._week_start_equity - current_equity) / self._week_start_equity
            if wdd >= self.weekly_loss_limit:
                reasons.append(f"weekly DD {wdd*100:.1f}%>={self.weekly_loss_limit*100:.0f}%")
        if self._consecutive_losses >= self.max_consecutive_losses:
            reasons.append(f"{self._consecutive_losses} consecutive losses")
        if reasons:
            logger.warning("KILL SWITCH: %s", "; ".join(reasons))
            self.halted = True
            return True
        return False

    # ------------------------------------------------------------------ #
    # Pre-trade gate + sizing
    # ------------------------------------------------------------------ #
    def pre_trade_check(
        self,
        score: float,
        rr: float,
        open_position_count: int,
        candidate_corr: Optional[float] = None,
        current_equity: Optional[float] = None,
    ) -> PreTradeDecision:
        d = PreTradeDecision(allowed=True)
        if self.halted:
            return PreTradeDecision(False, reasons=["trading halted by kill switch"])
        if current_equity is not None and self.kill_switch_triggered(current_equity):
            return PreTradeDecision(False, reasons=["kill switch"])
        if score < self.min_score:
            d.allowed = False; d.reasons.append(f"score {score:.0f}<{self.min_score:.0f}")
        if rr < self.min_rr:
            d.allowed = False; d.reasons.append(f"RR {rr:.1f}<{self.min_rr:.1f}")
        if open_position_count >= self.max_positions:
            d.allowed = False; d.reasons.append(f"max positions {open_position_count}/{self.max_positions}")
        if candidate_corr is not None and abs(candidate_corr) > self.max_correlation:
            d.allowed = False; d.reasons.append(f"corr {candidate_corr:.2f}>{self.max_correlation:.2f}")
        d.risk_fraction = self.risk_fraction_for_score(score) if d.allowed else 0.0
        return d

    def risk_fraction_for_score(self, score: float) -> float:
        """Base 1%, up to 2% for high conviction, down to 0.5% near the gate."""
        if score >= 85:
            return 0.02
        if score >= 75:
            return 0.01
        if score >= self.min_score:
            return 0.005
        return 0.0

    def adjust_fraction(
        self,
        fraction: float,
        volatility_regime: Optional[str] = None,
        candidate_corr: Optional[float] = None,
    ) -> float:
        """Scale the risk fraction down for high vol / correlated exposure."""
        adj = fraction
        if volatility_regime == "extreme":
            adj *= 0.5
        elif volatility_regime == "high":
            adj *= 0.75
        if candidate_corr is not None and abs(candidate_corr) > 0.5:
            adj *= max(0.5, 1 - (abs(candidate_corr) - 0.5))
        return adj

    # ------------------------------------------------------------------ #
    # Portfolio heat
    # ------------------------------------------------------------------ #
    def portfolio_heat(self, open_risk_dollars: Iterable[float], equity: float) -> float:
        if equity <= 0:
            return 0.0
        return float(sum(open_risk_dollars)) / equity

    def heat_allows(self, open_risk_dollars: Iterable[float], new_risk: float, equity: float) -> bool:
        if equity <= 0:
            return False
        total = (sum(open_risk_dollars) + new_risk) / equity
        if total > self.portfolio_heat_max:
            logger.info("Portfolio heat %.1f%% would exceed cap %.1f%%",
                        total * 100, self.portfolio_heat_max * 100)
            return False
        return True

    # ------------------------------------------------------------------ #
    # Correlation / exposure helpers
    # ------------------------------------------------------------------ #
    def can_open_new(self, open_position_count: int) -> bool:
        return open_position_count < self.max_positions

    @staticmethod
    def gross_exposure(positions: Iterable, equity: float) -> float:
        if equity <= 0:
            return 0.0
        total = 0.0
        for p in positions:
            try:
                total += abs(float(getattr(p, "market_value", 0.0) or 0.0))
            except (TypeError, ValueError):
                continue
        return total / equity

    def correlation_ok(self, candidate_closes: pd.Series, held_closes: dict[str, pd.Series]) -> bool:
        return self.max_candidate_correlation(candidate_closes, held_closes) <= self.max_correlation

    def max_candidate_correlation(
        self, candidate_closes: pd.Series, held_closes: dict[str, pd.Series]
    ) -> float:
        """Largest |corr| of the candidate's returns vs any held symbol (0 if none)."""
        try:
            if not held_closes or candidate_closes is None or len(candidate_closes) < 3:
                return 0.0
            cand = candidate_closes.pct_change().dropna()
            worst = 0.0
            for closes in held_closes.values():
                if closes is None or len(closes) < 3:
                    continue
                joined = pd.concat([cand, closes.pct_change().dropna()], axis=1, join="inner").dropna()
                if len(joined) < 3:
                    continue
                corr = joined.iloc[:, 0].corr(joined.iloc[:, 1])
                if corr is not None and np.isfinite(corr):
                    worst = max(worst, abs(corr))
            return float(worst)
        except Exception:
            logger.exception("max_candidate_correlation failed")
            return 0.0

    # ------------------------------------------------------------------ #
    # Dynamic stop management
    # ------------------------------------------------------------------ #
    def manage_stop(
        self, side: str, entry: float, current_price: float, current_stop: float,
        risk_per_share: float, atr: Optional[float] = None,
    ) -> float:
        """Return an updated stop: breakeven after 2R, ATR-trail after 3R.

        Never loosens an existing stop.
        """
        if risk_per_share <= 0:
            return current_stop
        if side == "long":
            r_mult = (current_price - entry) / risk_per_share
            new_stop = current_stop
            if r_mult >= 3 and atr:
                new_stop = max(new_stop, current_price - atr)   # trail
            elif r_mult >= 2:
                new_stop = max(new_stop, entry)                 # breakeven
            return new_stop
        else:  # short
            r_mult = (entry - current_price) / risk_per_share
            new_stop = current_stop
            if r_mult >= 3 and atr:
                new_stop = min(new_stop, current_price + atr)
            elif r_mult >= 2:
                new_stop = min(new_stop, entry)
            return new_stop
