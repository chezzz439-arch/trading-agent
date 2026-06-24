"""Regression tests for the options × portfolio-accounting interaction.

These cover the scan-ordering double-loss-booking holes an adversarial audit
found (2026-06-23): _detect_closed_trades runs BEFORE _manage_options, and used
to track option OCC symbols, so an organic option close (or a HALT flatten) got
re-booked as a fresh consecutive loss the next scan. Options must never be
tracked by _detect_closed_trades — they have their own lifecycle manager.

We exercise the REAL TradingAgent._detect_closed_trades via an unbound call with
a minimal fake `self`, so the actual production code path is tested.
"""

from __future__ import annotations

import types

from alpaca.trading.enums import AssetClass

import main
from src.signals.options_strategy import is_option_asset


def test_is_option_asset_handles_enum_and_string():
    # The real Alpaca attribute is the AssetClass enum, whose str() is
    # "AssetClass.US_OPTION" — the naive .endswith("us_option") was always False.
    assert is_option_asset(AssetClass.US_OPTION) is True
    assert is_option_asset(AssetClass.US_EQUITY) is False
    assert is_option_asset("us_option") is True          # raw-string form
    assert is_option_asset("us_equity") is False
    assert is_option_asset("") is False


class _Portfolio:
    def __init__(self):
        self.booked = []          # pnl values passed to record_trade_result

    def record_trade_result(self, pnl):
        self.booked.append(pnl)


class _Notifier:
    def __init__(self):
        self.closed = []

    def trade_closed(self, **kw):
        self.closed.append(kw)


def _fake_self():
    s = types.SimpleNamespace()
    s.managed = {}
    s._closed_by_manager = set()
    s._kill_switch_closed = set()
    s._pos_pnl = {}
    s._open_risk = {}
    s._closed_today = []
    s.portfolio = _Portfolio()
    s.notifier = _Notifier()
    return s


def _pos(symbol, asset_class, upl):
    return types.SimpleNamespace(symbol=symbol, asset_class=asset_class,
                                 unrealized_pl=upl)


def test_option_position_is_never_tracked_for_closed_detection():
    s = _fake_self()
    positions = [
        _pos("MSFT", AssetClass.US_EQUITY, 12.0),
        _pos("AAL260717C00016000", AssetClass.US_OPTION, -130.0),
    ]
    main.TradingAgent._detect_closed_trades(s, positions, equity=100_000)
    # The equity is tracked; the option OCC symbol is NOT.
    assert "MSFT" in s._pos_pnl
    assert "AAL260717C00016000" not in s._pos_pnl


def test_vanished_option_is_not_rebooked_as_loss():
    # Simulate the dangerous sequence: an option was (wrongly) in _pos_pnl, then
    # vanishes. With the fix it can't be in _pos_pnl, but even if a stale entry
    # existed, a position list containing only the equity must not book it twice.
    s = _fake_self()
    s._pos_pnl = {"MSFT": 5.0}           # only the equity was ever tracked
    # Option closed by _manage_options last scan; broker now reports only MSFT.
    positions = [_pos("MSFT", AssetClass.US_EQUITY, 7.0)]
    main.TradingAgent._detect_closed_trades(s, positions, equity=100_000)
    assert s.portfolio.booked == []      # nothing closed → nothing booked
    assert s._pos_pnl == {"MSFT": 7.0}


def test_equity_close_is_still_booked_once():
    # Baseline: a genuine equity close (not in any exclusion set) books exactly once.
    s = _fake_self()
    s._pos_pnl = {"TSLA": -200.0}
    positions = []                        # TSLA gone, not manager/kill-switch closed
    main.TradingAgent._detect_closed_trades(s, positions, equity=100_000)
    assert s.portfolio.booked == [-200.0]
    assert len(s.notifier.closed) == 1


def test_kill_switch_closed_equity_not_rebooked():
    # An equity flattened by the kill switch (seeded into _kill_switch_closed) is
    # consumed once and never booked — covers the HALT-button seeding fix too.
    s = _fake_self()
    s._pos_pnl = {"TSLA": -200.0}
    s._kill_switch_closed = {"TSLA"}
    positions = []
    main.TradingAgent._detect_closed_trades(s, positions, equity=100_000)
    assert s.portfolio.booked == []
    assert "TSLA" not in s._kill_switch_closed   # consumed
