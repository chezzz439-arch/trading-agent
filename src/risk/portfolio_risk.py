"""Portfolio-level risk controls.

Enforces, across the whole account:

* a hard cap on concurrent positions (default 3),
* a daily-loss kill switch (default -3% from the day's starting equity),
* a correlation gate so new positions aren't too correlated with open ones,
* a view of total gross exposure relative to equity.

All checks are defensive and never raise on bad inputs.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PortfolioRisk:
    def __init__(
        self,
        max_positions: int = 3,
        daily_loss_limit: float = 0.03,
        max_correlation: float = 0.80,
    ) -> None:
        self.max_positions = max_positions
        self.daily_loss_limit = daily_loss_limit
        self.max_correlation = max_correlation
        self._day_start_equity: Optional[float] = None

    # ------------------------------------------------------------------ #
    # Daily loss kill switch
    # ------------------------------------------------------------------ #
    def set_day_start_equity(self, equity: float) -> None:
        """Record the equity baseline the kill switch measures against."""
        self._day_start_equity = equity
        logger.info("Day-start equity set to $%.2f", equity)

    def kill_switch_triggered(self, current_equity: float) -> bool:
        """True once drawdown from the day's start reaches the loss limit."""
        if self._day_start_equity is None or self._day_start_equity <= 0:
            return False
        drawdown = (self._day_start_equity - current_equity) / self._day_start_equity
        if drawdown >= self.daily_loss_limit:
            logger.warning(
                "KILL SWITCH: drawdown %.2f%% >= limit %.2f%% "
                "(start=$%.2f now=$%.2f)",
                drawdown * 100, self.daily_loss_limit * 100,
                self._day_start_equity, current_equity,
            )
            return True
        return False

    # ------------------------------------------------------------------ #
    # Position-count gate
    # ------------------------------------------------------------------ #
    def can_open_new(self, open_position_count: int) -> bool:
        if open_position_count >= self.max_positions:
            logger.info(
                "Max concurrent positions reached (%d/%d)",
                open_position_count, self.max_positions,
            )
            return False
        return True

    # ------------------------------------------------------------------ #
    # Exposure
    # ------------------------------------------------------------------ #
    @staticmethod
    def gross_exposure(positions: Iterable, equity: float) -> float:
        """Sum of |market value| of positions divided by equity."""
        if equity <= 0:
            return 0.0
        total = 0.0
        for p in positions:
            try:
                total += abs(float(getattr(p, "market_value", 0.0) or 0.0))
            except (TypeError, ValueError):
                continue
        return total / equity

    # ------------------------------------------------------------------ #
    # Correlation gate
    # ------------------------------------------------------------------ #
    def correlation_ok(
        self,
        candidate_closes: pd.Series,
        held_closes: dict[str, pd.Series],
    ) -> bool:
        """True if the candidate isn't over-correlated with any held symbol.

        Uses pairwise correlation of daily returns over the overlapping window.
        Missing/insufficient data is treated as "ok" (fails open) so the check
        never blocks trading purely due to a data gap.
        """
        try:
            if not held_closes or candidate_closes is None or len(candidate_closes) < 3:
                return True
            cand_ret = candidate_closes.pct_change().dropna()
            for symbol, closes in held_closes.items():
                if closes is None or len(closes) < 3:
                    continue
                held_ret = closes.pct_change().dropna()
                joined = pd.concat([cand_ret, held_ret], axis=1, join="inner").dropna()
                if len(joined) < 3:
                    continue
                corr = joined.iloc[:, 0].corr(joined.iloc[:, 1])
                if corr is not None and np.isfinite(corr) and abs(corr) >= self.max_correlation:
                    logger.info(
                        "Correlation gate: candidate vs %s = %.2f >= %.2f",
                        symbol, corr, self.max_correlation,
                    )
                    return False
            return True
        except Exception:
            logger.exception("correlation_ok failed; allowing trade")
            return True
