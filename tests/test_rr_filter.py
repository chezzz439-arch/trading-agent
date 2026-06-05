"""Unit tests for the RR filter, ATR calculation, and position sizing math."""

from __future__ import annotations

import pandas as pd
import pytest

from src.risk.position_sizer import PositionSizer
from src.signals.rr_filter import RRFilter, TradePlan, atr
from src.signals.strategy import Signal


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
def make_df(highs, lows, closes) -> pd.DataFrame:
    n = len(closes)
    idx = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {"open": closes, "high": highs, "low": lows, "close": closes,
         "volume": [1_000.0] * n},
        index=idx,
    )


def constant_range_df(n: int = 130, high: float = 101.0, low: float = 99.0,
                      close: float = 100.0) -> pd.DataFrame:
    """Every bar has the same range, so ATR == (high - low) exactly."""
    return make_df([high] * n, [low] * n, [close] * n)


def clean_long_df(n: int = 130) -> pd.DataFrame:
    """Bars whose highs sit below the (entry=100) signal — a blue-sky breakout.

    Constant 1.0 range => ATR == 1.0, so stop distance = 1.5 and the constructed
    5:1 target sits at 100 + 5 * 1.5 = 107.5.
    """
    return make_df([99.5] * n, [98.5] * n, [99.0] * n)


def long_signal(df: pd.DataFrame, entry: float = 100.0) -> Signal:
    return Signal("TEST", "long", entry_price=entry, rsi=60.0,
                  timestamp=df.index[-1], reason="test")


# --------------------------------------------------------------------------- #
# ATR
# --------------------------------------------------------------------------- #
def test_atr_constant_range_equals_range():
    # With a constant 2.0 bar range and no gaps, true range is 2.0 every bar,
    # so Wilder-smoothed ATR is exactly 2.0.
    df = constant_range_df(high=101.0, low=99.0, close=100.0)
    assert atr(df, period=14).iloc[-1] == pytest.approx(2.0, abs=1e-9)


def test_atr_is_positive_and_finite():
    df = constant_range_df()
    series = atr(df, 14)
    assert (series > 0).all()
    assert series.notna().all()


# --------------------------------------------------------------------------- #
# RR filter — constructed target + structural path veto
# --------------------------------------------------------------------------- #
def test_constructed_target_is_exactly_5to1():
    # Blue-sky breakout: ATR 1.0 -> stop 98.5, target 100 + 5*1.5 = 107.5.
    df = clean_long_df()
    plan = RRFilter(rr_ratio=5.0, atr_multiplier=1.5).evaluate(long_signal(df), df)
    assert plan is not None
    assert plan.rr == pytest.approx(5.0, abs=1e-6)
    assert plan.stop < plan.entry < plan.target
    assert plan.stop == pytest.approx(98.5, abs=1e-6)
    assert plan.target == pytest.approx(107.5, abs=1e-6)


def test_resistance_in_path_is_rejected():
    # Park a swing high (103) between entry (100) and the ~107.5 target: the
    # structural veto should block the trade.
    df = clean_long_df()
    resistance = 103.0
    df.iloc[60, df.columns.get_loc("high")] = resistance

    a = float(atr(df, 14).iloc[-1])
    target = 100.0 + 5.0 * 1.5 * a
    assert 100.0 < resistance < target  # precondition: resistance sits in path

    plan = RRFilter(rr_ratio=5.0, atr_multiplier=1.5).evaluate(long_signal(df), df)
    assert plan is None


def test_resistance_beyond_target_passes():
    # A swing high well above the target is not "in the path" — trade allowed.
    df = clean_long_df()
    resistance = 130.0
    df.iloc[40, df.columns.get_loc("high")] = resistance

    a = float(atr(df, 14).iloc[-1])
    target = 100.0 + 5.0 * 1.5 * a
    assert resistance >= target  # precondition: resistance is beyond the target

    plan = RRFilter(rr_ratio=5.0, atr_multiplier=1.5).evaluate(long_signal(df), df)
    assert plan is not None
    # rr is computed from penny-rounded levels, so allow minor rounding drift.
    assert plan.rr == pytest.approx(5.0, abs=0.1)


def test_blue_sky_breakout_passes():
    # No overhead resistance at all (all highs below entry) -> clear path.
    df = clean_long_df()
    plan = RRFilter().evaluate(long_signal(df, entry=100.0), df)
    assert plan is not None


# --------------------------------------------------------------------------- #
# Position sizing
# --------------------------------------------------------------------------- #
def _plan(entry, stop, target) -> TradePlan:
    return TradePlan(
        symbol="TEST", side="long", entry=entry, stop=stop, target=target,
        rr=round(abs(target - entry) / abs(entry - stop), 2),
        risk_per_share=abs(entry - stop), atr=2.0, reason="test",
    )


def test_position_size_risk_path():
    # entry/risk = 100/20 = 5 < 10, so the 10% notional cap does NOT bind.
    # risk_budget = 1% * 100k = $1,000 ; qty = 1000 / 20 = 50.
    sizer = PositionSizer(risk_per_trade=0.01, max_position_pct=0.10)
    trade = sizer.size(_plan(entry=100.0, stop=80.0, target=200.0), equity=100_000)
    assert trade is not None
    assert trade.qty == 50
    assert trade.dollar_risk == pytest.approx(1_000.0)   # exactly 1% of equity
    assert trade.dollar_target == pytest.approx(5_000.0)  # 50 * (200 - 100)


def test_position_size_capped_by_max_position():
    # risk_budget = $1,000 / $3 risk = 333 shares, but 10% of $100k = $10k cap
    # at $100 entry => max 100 shares. Cap binds.
    sizer = PositionSizer(risk_per_trade=0.01, max_position_pct=0.10)
    trade = sizer.size(_plan(entry=100.0, stop=97.0, target=115.0), equity=100_000)
    assert trade is not None
    assert trade.qty == 100
    assert trade.dollar_risk == pytest.approx(300.0)
    assert trade.dollar_target == pytest.approx(1_500.0)


def test_position_size_fractional_crypto():
    # Crypto allows fractional qty; the 10% cap binds: $10k / $30k = 0.3333 BTC.
    sizer = PositionSizer(risk_per_trade=0.01, max_position_pct=0.10)
    trade = sizer.size(
        _plan(entry=30_000.0, stop=29_000.0, target=35_000.0),
        equity=100_000, fractional=True,
    )
    assert trade is not None
    assert 0 < trade.qty < 1
    assert trade.qty == pytest.approx(0.333333, abs=1e-6)


def test_zero_risk_returns_none():
    sizer = PositionSizer()
    plan = TradePlan("TEST", "long", 100.0, 100.0, 115.0, 0.0, 0.0, 2.0, "test")
    assert sizer.size(plan, equity=100_000) is None
