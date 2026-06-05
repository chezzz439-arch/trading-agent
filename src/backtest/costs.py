"""Transaction-cost model for the backtester.

Applies two frictions that separate a paper backtest from reality:

* **Slippage** — each fill is worsened by ``slippage_bps`` of price (buys fill
  higher, sells fill lower), on both entry and exit.
* **Commission** — a flat per-fill charge plus a bps-of-notional charge.

Defaults are **zero** so existing cost-free backtests are unchanged; the
validation harness constructs a realistic model explicitly (e.g.
``CostModel.equities()`` / ``CostModel.crypto()``).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    slippage_bps: float = 0.0          # adverse slippage per fill, in basis points
    commission_per_trade: float = 0.0  # flat $ per fill
    commission_bps: float = 0.0        # bps of notional per fill

    # -- presets ------------------------------------------------------- #
    @classmethod
    def equities(cls) -> "CostModel":
        # Alpaca equities are commission-free; model spread/impact as slippage.
        return cls(slippage_bps=2.0, commission_per_trade=0.0, commission_bps=0.0)

    @classmethod
    def crypto(cls) -> "CostModel":
        # Wider spreads + ~15-25 bps taker fees on crypto venues.
        return cls(slippage_bps=8.0, commission_per_trade=0.0, commission_bps=20.0)

    # -- application --------------------------------------------------- #
    def fill_price(self, price: float, side: str, is_entry: bool) -> float:
        """Worsen ``price`` by slippage in the adverse direction for this fill."""
        slip = price * self.slippage_bps / 1e4
        if is_entry:
            adverse = 1 if side == "long" else -1   # entering long pays up
        else:
            adverse = -1 if side == "long" else 1   # exiting long sells down
        return price + adverse * slip

    def commission(self, qty: float, price: float) -> float:
        return self.commission_per_trade + abs(qty * price) * self.commission_bps / 1e4

    def round_trip_commission(self, qty: float, entry: float, exit_price: float) -> float:
        return self.commission(qty, entry) + self.commission(qty, exit_price)

    @property
    def is_zero(self) -> bool:
        return (self.slippage_bps == 0 and self.commission_per_trade == 0
                and self.commission_bps == 0)
