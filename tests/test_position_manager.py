"""Tests for the stateful P9 position manager decision engine + persistence."""

from __future__ import annotations

import pytest

from src.execution.position_manager import (
    ManagedPosition,
    PositionManager,
    PositionStore,
)


def make_pos(side="long", entry=100.0, stop=98.0, target=110.0, qty=300,
             fractional=False) -> ManagedPosition:
    # risk/share = 2.0  =>  R levels at 2,3.5,5 etc.
    return ManagedPosition(
        symbol="TEST", side=side, entry=entry, initial_stop=stop, current_stop=stop,
        target=target, risk_per_share=abs(entry - stop), initial_qty=qty,
        remaining_qty=qty, atr=2.0, fractional=fractional,
    )


def kinds(actions):
    return [(a.kind, a.tag) for a in actions]


# --------------------------------------------------------------------------- #
# Scale-outs
# --------------------------------------------------------------------------- #
def test_scale_out_at_2r():
    mp = make_pos()
    pm = PositionManager()
    acts = pm.update(mp, price=104.0)        # +2R (risk/share 2)
    assert ("scale_out", "2R") in kinds(acts)
    assert mp.remaining_qty == 300 - 99      # floor(300*0.33)=99 taken
    assert mp.realized_pnl == pytest.approx(99 * (104 - 100))


def test_scale_out_at_3p5r_then_target():
    mp = make_pos()
    pm = PositionManager()
    pm.update(mp, price=104.0)               # 2R tranche
    pm.update(mp, price=107.0)               # +3.5R tranche
    assert "3.5R" in mp.tranches_taken
    # Remaining rides to target; hitting it closes the rest.
    acts = pm.update(mp, price=110.0)
    assert ("close_hit", "target") in kinds(acts)
    assert mp.status == "closed" and mp.remaining_qty == 0
    # Realized PnL is exact across all tranches.
    assert mp.realized_pnl == pytest.approx(99 * 4 + 99 * 7 + 102 * 10)


# --------------------------------------------------------------------------- #
# Dynamic stops
# --------------------------------------------------------------------------- #
def test_breakeven_at_2r():
    mp = make_pos()
    pm = PositionManager()
    acts = pm.update(mp, price=104.0)        # +2R
    assert ("move_stop", "breakeven") in kinds(acts)
    assert mp.current_stop == mp.entry and mp.breakeven_done


def test_trailing_from_3r():
    mp = make_pos()
    pm = PositionManager()
    acts = pm.update(mp, price=106.0, atr=2.0)   # +3R -> trail to 106-2=104
    assert ("move_stop", "trail") in kinds(acts)
    assert mp.current_stop == pytest.approx(104.0) and mp.trailing_active


def test_stop_never_loosens():
    mp = make_pos()
    pm = PositionManager()
    pm.update(mp, price=106.0, atr=2.0)      # trail to 104
    before = mp.current_stop
    pm.update(mp, price=104.5, atr=2.0, advance_bar=True)  # pulled back; no looser stop
    assert mp.current_stop >= before


def test_trailing_stop_gets_hit_locks_profit():
    mp = make_pos()
    pm = PositionManager()
    pm.update(mp, price=106.0, atr=2.0)      # +3R: scales 2R tranche @106, trails to 104
    acts = pm.update(mp, price=103.9, atr=2.0)   # drops below trailing stop
    assert ("close_hit", "stop") in kinds(acts)
    assert mp.status == "closed"
    # 99 sh scaled at +6, remaining 201 sh stopped out at the +2R trail (+4): all profit.
    assert mp.realized_pnl == pytest.approx(99 * 6 + 201 * 4)
    assert mp.realized_pnl > 0


# --------------------------------------------------------------------------- #
# Time exit + protective stop
# --------------------------------------------------------------------------- #
def test_time_exit_when_stalled():
    mp = make_pos()
    pm = PositionManager(time_exit_bars=10, time_exit_min_r=1.0)
    last = []
    for _ in range(10):                      # 10 bars hovering at ~+0.5R
        last = pm.update(mp, price=101.0, atr=2.0)
    assert ("time_exit", "time") in kinds(last)
    assert mp.status == "closed"


def test_scan_cycles_do_not_count_as_bars():
    """Intra-day scan cycles (advance_bar=False) must NOT age the time-exit
    clock — only real daily bars do. Guards the unit mismatch that force-closed
    positions ~40 min after entry instead of after time_exit_bars *days*."""
    mp = make_pos()
    pm = PositionManager(time_exit_bars=10, time_exit_min_r=1.0)
    for _ in range(200):                     # 200 cycles in a day, stalled at +0.5R
        acts = pm.update(mp, price=101.0, atr=2.0, advance_bar=False)
        assert ("time_exit", "time") not in kinds(acts)
    assert mp.bars_held == 0 and mp.status == "open"
    # Now advance 10 genuine daily bars -> time-exit fires.
    last = []
    for _ in range(10):
        last = pm.update(mp, price=101.0, atr=2.0, advance_bar=True)
    assert mp.bars_held == 10
    assert ("time_exit", "time") in kinds(last) and mp.status == "closed"


def test_initial_stop_hit_is_a_loss():
    mp = make_pos()
    pm = PositionManager()
    acts = pm.update(mp, price=97.9)         # below 98 stop
    assert ("close_hit", "stop") in kinds(acts)
    assert mp.realized_pnl == pytest.approx(300 * (98 - 100))   # -1R full size


# --------------------------------------------------------------------------- #
# Short side mirror
# --------------------------------------------------------------------------- #
def test_short_scale_and_breakeven():
    mp = make_pos(side="short", entry=100.0, stop=102.0, target=90.0)
    pm = PositionManager()
    acts = pm.update(mp, price=96.0)         # +2R for a short
    assert ("scale_out", "2R") in kinds(acts)
    assert ("move_stop", "breakeven") in kinds(acts)
    assert mp.realized_pnl == pytest.approx(99 * (100 - 96))


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def test_position_store_roundtrip(tmp_path):
    store = PositionStore(log_dir=str(tmp_path))
    mp = make_pos()
    mp.bars_held = 3
    mp.tranches_taken.append("2R")
    store.save({"TEST": mp})
    loaded = store.load()
    assert "TEST" in loaded
    r = loaded["TEST"]
    assert r.bars_held == 3 and r.tranches_taken == ["2R"] and r.entry == 100.0


def test_position_store_drops_closed(tmp_path):
    store = PositionStore(log_dir=str(tmp_path))
    mp = make_pos()
    mp.status = "closed"
    store.save({"TEST": mp})
    assert store.load() == {}    # closed positions are not persisted
