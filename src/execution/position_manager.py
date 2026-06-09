"""Phase 9 (stateful) — position lifecycle manager.

Tracks each open position from entry to full close and drives its lifecycle:

* **Scale-out** — take a tranche at +2R and another at +3.5R; the remainder
  rides to the target.
* **Dynamic stop** — move to breakeven at +2R, then ATR-trail from +3R.
* **Time exit** — close a position that hasn't reached +1R after N bars.
* **Exact realized PnL** — accumulated across tranches, so trade-closed reporting
  is precise rather than estimated.

``PositionManager.update`` is a **pure decision engine**: given a position's
state plus the current price (and ATR), it mutates the state and returns a list
of :class:`Action`s for an executor to carry out. This is fully unit-testable
and identical in backtest and live; only the executor (broker calls) differs.

``PositionStore`` persists open positions to JSON so lifecycle state survives an
agent restart.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ============================================================================ #
# Actions (what the executor should do)
# ============================================================================ #
@dataclass
class Action:
    kind: str          # scale_out | move_stop | time_exit | close_hit
    symbol: str
    qty: float = 0.0
    price: float = 0.0
    new_stop: float = 0.0
    tag: str = ""      # 2R | 3.5R | breakeven | trail | stop | target | time
    realized_pnl: float = 0.0
    note: str = ""


# ============================================================================ #
# Managed position state
# ============================================================================ #
@dataclass
class ManagedPosition:
    symbol: str
    side: str                  # long | short
    entry: float
    initial_stop: float
    current_stop: float
    target: float
    risk_per_share: float
    initial_qty: float
    remaining_qty: float
    atr: float = 0.0
    score: float = 0.0
    regime: str = ""           # regime label at entry
    fractional: bool = False
    entry_time: str = ""
    bars_held: int = 0         # number of *daily* bars held (not scan cycles)
    last_bar_date: str = ""    # date of the last counted daily bar; gates time-exit
    tranches_taken: list = field(default_factory=list)
    breakeven_done: bool = False
    trailing_active: bool = False
    realized_pnl: float = 0.0
    last_r: float = 0.0
    confirmed: bool = False    # True once the fill is seen at the broker
    status: str = "open"       # open | closed

    @classmethod
    def from_trade(cls, trade, *, score: float = 0.0, atr: float = 0.0,
                   regime: str = "", fractional: bool = False,
                   entry_time: str = "") -> "ManagedPosition":
        p = trade.plan
        return cls(
            symbol=p.symbol, side=p.side, entry=p.entry, initial_stop=p.stop,
            current_stop=p.stop, target=p.target, risk_per_share=p.risk_per_share,
            initial_qty=trade.qty, remaining_qty=trade.qty, atr=atr, score=score,
            regime=regime, fractional=fractional, entry_time=entry_time,
        )

    @property
    def realized_r(self) -> float:
        risk = self.initial_qty * self.risk_per_share
        return self.realized_pnl / risk if risk else 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ManagedPosition":
        return cls(**d)


# ============================================================================ #
# Decision engine
# ============================================================================ #
class PositionManager:
    def __init__(
        self,
        scale1_r: float = 2.0,
        scale2_r: float = 3.5,
        scale_fraction: float = 0.33,
        breakeven_r: float = 2.0,
        trail_r: float = 3.0,
        time_exit_bars: int = 10,
        time_exit_min_r: float = 1.0,
    ) -> None:
        self.scale1_r = scale1_r
        self.scale2_r = scale2_r
        self.scale_fraction = scale_fraction
        self.breakeven_r = breakeven_r
        self.trail_r = trail_r
        self.time_exit_bars = time_exit_bars
        self.time_exit_min_r = time_exit_min_r

    def update(self, mp: ManagedPosition, price: float,
               atr: Optional[float] = None, advance_bar: bool = True) -> list[Action]:
        """Advance one tick: mutate ``mp`` and return the actions to execute."""
        actions: list[Action] = []
        if mp.status != "open":
            return actions
        if advance_bar:
            mp.bars_held += 1
        d = 1 if mp.side == "long" else -1
        mp.last_r = d * (price - mp.entry) / mp.risk_per_share if mp.risk_per_share else 0.0

        # 1) Hard exits: protective stop or final target hit on the remainder.
        if self._stop_hit(mp, price):
            actions.append(self._close_remaining(mp, mp.current_stop, "stop"))
            return actions
        if self._target_hit(mp, price):
            actions.append(self._close_remaining(mp, mp.target, "target"))
            return actions

        # 2) Scale-outs at +2R and +3.5R.
        for level, tag in ((self.scale1_r, "2R"), (self.scale2_r, "3.5R")):
            if mp.last_r >= level and tag not in mp.tranches_taken and mp.remaining_qty > 0:
                qty = self._tranche_qty(mp)
                if qty <= 0 or qty > mp.remaining_qty:
                    mp.tranches_taken.append(tag)   # too small to split; skip cleanly
                    continue
                pnl = d * (price - mp.entry) * qty
                mp.remaining_qty = self._round_qty(mp.remaining_qty - qty, mp.fractional)
                mp.realized_pnl += pnl
                mp.tranches_taken.append(tag)
                actions.append(Action("scale_out", mp.symbol, qty=qty, price=price,
                                      tag=tag, realized_pnl=pnl,
                                      note=f"scaled out {tag}"))
                if mp.remaining_qty <= 0:
                    mp.status = "closed"
                    return actions

        # 3) Stop management: trail from +3R, else breakeven from +2R.
        res = self._manage_stop(mp, price, atr)
        if res is not None:
            new_stop, kind = res
            mp.current_stop = new_stop
            if kind == "breakeven":
                mp.breakeven_done = True
            else:
                mp.trailing_active = True
            actions.append(Action("move_stop", mp.symbol, new_stop=new_stop,
                                  price=price, tag=kind))

        # 4) Time-based exit for a stalled position.
        if mp.bars_held >= self.time_exit_bars and mp.last_r < self.time_exit_min_r:
            actions.append(self._close_remaining(mp, price, "time"))

        return actions

    # ------------------------------------------------------------------ #
    def _stop_hit(self, mp, price) -> bool:
        return price <= mp.current_stop if mp.side == "long" else price >= mp.current_stop

    def _target_hit(self, mp, price) -> bool:
        return price >= mp.target if mp.side == "long" else price <= mp.target

    def _tranche_qty(self, mp) -> float:
        raw = mp.initial_qty * self.scale_fraction
        return round(raw, 6) if mp.fractional else math.floor(raw)

    @staticmethod
    def _round_qty(qty, fractional) -> float:
        return round(qty, 6) if fractional else int(round(qty))

    def _tighter(self, mp, candidate) -> bool:
        return candidate > mp.current_stop if mp.side == "long" else candidate < mp.current_stop

    def _manage_stop(self, mp, price, atr):
        """Return (new_stop, kind) if the stop should tighten, else None."""
        d = 1 if mp.side == "long" else -1
        r = d * (price - mp.entry) / mp.risk_per_share if mp.risk_per_share else 0.0
        # Trail from +3R using one ATR.
        if r >= self.trail_r and atr:
            candidate = price - atr if mp.side == "long" else price + atr
            if self._tighter(mp, candidate):
                return candidate, "trail"
        # Breakeven from +2R (only once).
        if r >= self.breakeven_r and not mp.breakeven_done and self._tighter(mp, mp.entry):
            return mp.entry, "breakeven"
        return None

    def _close_remaining(self, mp, price, tag) -> Action:
        d = 1 if mp.side == "long" else -1
        qty = mp.remaining_qty
        pnl = d * (price - mp.entry) * qty
        mp.realized_pnl += pnl
        mp.remaining_qty = 0
        mp.status = "closed"
        kind = "time_exit" if tag == "time" else "close_hit"
        return Action(kind, mp.symbol, qty=qty, price=price, tag=tag,
                      realized_pnl=pnl, note=f"closed remainder ({tag})")


# ============================================================================ #
# Persistence
# ============================================================================ #
class PositionStore:
    def __init__(self, log_dir: str = "logs"):
        self.path = os.path.join(log_dir, "positions.json")
        os.makedirs(log_dir, exist_ok=True)

    def save(self, managed: dict[str, ManagedPosition]) -> None:
        try:
            data = {sym: mp.to_dict() for sym, mp in managed.items()
                    if mp.status == "open"}
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, default=str)
            os.replace(tmp, self.path)
        except Exception:
            logger.exception("PositionStore.save failed")

    def load(self) -> dict[str, ManagedPosition]:
        try:
            with open(self.path) as f:
                data = json.load(f)
            return {sym: ManagedPosition.from_dict(d) for sym, d in data.items()}
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
        except Exception:
            logger.exception("PositionStore.load failed")
            return {}
