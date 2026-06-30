"""Fixed-fractional position sizing.

Risk a constant fraction of equity (default 1%) per trade, then cap the
position so its notional never exceeds ``max_position_pct`` of equity
(default 10%). Stocks are sized in whole shares; crypto allows fractional qty.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from src.signals.rr_filter import TradePlan

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SizedTrade:
    plan: TradePlan
    qty: float            # whole shares for stocks, fractional for crypto
    dollar_risk: float    # qty * risk_per_share
    dollar_target: float  # qty * reward_per_share (profit if target hit)
    notional: float       # qty * entry


class PositionSizer:
    def __init__(
        self,
        risk_per_trade: float = 0.01,
        max_position_pct: float = 0.10,
    ) -> None:
        if not 0 < risk_per_trade < 1:
            raise ValueError("risk_per_trade must be in (0, 1)")
        if not 0 < max_position_pct <= 1:
            raise ValueError("max_position_pct must be in (0, 1]")
        self.risk_per_trade = risk_per_trade
        self.max_position_pct = max_position_pct

    def size(
        self,
        plan: TradePlan,
        equity: float,
        fractional: bool = False,
    ) -> SizedTrade | None:
        """Return a SizedTrade, or None if no viable position.

        ``fractional=True`` (crypto) keeps fractional qty; otherwise qty is
        floored to whole shares.
        """
        try:
            # Reject non-finite inputs up front: a NaN/inf risk_per_share or entry
            # slips past the ``<= 0`` checks (``NaN <= 0`` is False) and later blows
            # up at ``math.floor(NaN)``, which is caught but logs a full traceback
            # every bad bar. Same safe outcome (None), without the noise.
            if (not math.isfinite(equity) or equity <= 0
                    or not math.isfinite(plan.risk_per_share) or plan.risk_per_share <= 0
                    or not math.isfinite(plan.entry) or plan.entry <= 0):
                return None

            # 1) Risk-based quantity: dollars at risk == risk_per_trade * equity.
            risk_budget = equity * self.risk_per_trade
            raw_qty = risk_budget / plan.risk_per_share

            # 2) Cap by max position notional.
            max_notional = equity * self.max_position_pct
            max_qty_by_notional = max_notional / plan.entry
            qty = min(raw_qty, max_qty_by_notional)

            if not fractional:
                qty = math.floor(qty)
            else:
                qty = math.floor(qty * 1e6) / 1e6  # 6 dp for crypto

            if qty <= 0:
                logger.info(
                    "%s: position rounds to zero (risk_budget=$%.2f, "
                    "risk/share=$%.4f)", plan.symbol, risk_budget, plan.risk_per_share,
                )
                return None

            reward_per_share = abs(plan.target - plan.entry)
            return SizedTrade(
                plan=plan,
                qty=qty,
                dollar_risk=round(qty * plan.risk_per_share, 2),
                dollar_target=round(qty * reward_per_share, 2),
                notional=round(qty * plan.entry, 2),
            )
        except Exception:
            logger.exception("PositionSizer.size failed for %s", plan.symbol)
            return None
