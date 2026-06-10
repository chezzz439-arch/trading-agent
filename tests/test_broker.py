"""Broker execution guards: bracket-band validation + price precision.

Regression coverage for the CRCL bug — a ranging-regime smart LIMIT entry used
an EMA pullback reference that landed *outside* the (target, stop) band, which
made the protective stop fall on the wrong side of the entry and Alpaca rejected
the whole bracket. The broker must now detect that and fall back to a market
bracket instead of erroring.
"""

from __future__ import annotations

import pytest

from src.execution.broker import Broker, _round_price
from src.risk.position_sizer import SizedTrade
from src.signals.rr_filter import TradePlan


class _FakeOrder:
    id = "FAKE"
    status = "accepted"


class _FakeClient:
    """Records the request type submitted so we can assert market vs limit."""

    def __init__(self):
        self.submitted = []

    def submit_order(self, order):
        self.submitted.append(type(order).__name__)
        return _FakeOrder()


def _broker():
    b = Broker.__new__(Broker)        # bypass real TradingClient construction
    b._client = _FakeClient()
    b.paper = True
    return b


def _trade(side, entry, stop, target, qty=10):
    plan = TradePlan("CRCL", side, entry, stop, target, 3.0, abs(entry - stop), 2.0, "test")
    return SizedTrade(plan, qty=qty, dollar_risk=qty * abs(entry - stop),
                      dollar_target=0.0, notional=qty * entry)


# --- band validation ----------------------------------------------------- #

def test_short_limit_above_stop_falls_back_to_market():
    """The CRCL case: short limit (EMA) above the stop -> invalid -> market bracket."""
    b = _broker()
    b._place_limit_bracket(_trade("short", 82.04, 85.21, 72.52), limit_price=91.34)
    assert b._client.submitted == ["MarketOrderRequest"]


def test_long_limit_above_target_falls_back_to_market():
    b = _broker()
    b._place_limit_bracket(_trade("long", 82.0, 78.0, 90.0), limit_price=95.0)
    assert b._client.submitted == ["MarketOrderRequest"]


def test_valid_short_pullback_uses_limit_bracket():
    b = _broker()
    b._place_limit_bracket(_trade("short", 82.04, 85.21, 72.52), limit_price=84.0)
    assert b._client.submitted == ["LimitOrderRequest"]


def test_valid_long_pullback_uses_limit_bracket():
    b = _broker()
    b._place_limit_bracket(_trade("long", 82.0, 78.0, 90.0), limit_price=80.0)
    assert b._client.submitted == ["LimitOrderRequest"]


# --- price precision ------------------------------------------------------ #

@pytest.mark.parametrize("raw,expected", [
    (78.123456, 78.12),    # >= $1 -> penny
    (90.987654, 90.99),
    (0.123456, 0.1235),    # < $1 -> sub-penny
    (1.0, 1.0),
])
def test_round_price_ticks(raw, expected):
    assert _round_price(raw) == expected


def test_limit_bracket_does_not_crash_on_subpenny_legs():
    b = _broker()
    b._place_limit_bracket(_trade("long", 82.0, 78.123456, 90.987654), limit_price=80.0)
    assert b._client.submitted == ["LimitOrderRequest"]
