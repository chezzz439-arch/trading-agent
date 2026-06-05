"""Tests for the per-symbol universe screener (no network — cache injected)."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.data.screener import ScreenCriteria, UniverseScreener


def make_df(close: float, volume: float, n: int = 30) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({"open": close, "high": close, "low": close,
                         "close": close, "volume": volume}, index=idx)


def _inject(s: UniverseScreener, symbol: str, **fund) -> None:
    s._cache[symbol] = (fund, datetime.now(timezone.utc))


def test_crypto_bypasses_equity_filters():
    ok, reason = UniverseScreener().passes("BTC/USD")
    assert ok and "crypto" in reason


def test_price_floor_rejects():
    s = UniverseScreener(ScreenCriteria(min_price=15))
    _inject(s, "LOW", market_cap=5e9, avg_volume=5e6, exchange="NMS")
    ok, reason = s.passes("LOW", make_df(close=9.0, volume=5e6))
    assert not ok and "price" in reason


def test_volume_floor_uses_full_market_volume_not_iex():
    # df shows huge IEX-style volume but the (full-market) fundamentals avg is
    # thin -> must reject on the fundamentals figure, not the df.
    s = UniverseScreener(ScreenCriteria(min_avg_volume=1e6))
    _inject(s, "THIN", market_cap=5e9, avg_volume=100_000, exchange="NMS")
    ok, reason = s.passes("THIN", make_df(close=50.0, volume=9e6))
    assert not ok and "vol" in reason


def test_market_cap_floor_rejects():
    s = UniverseScreener(ScreenCriteria(min_market_cap=3e9))
    _inject(s, "SMALL", market_cap=1e9, avg_volume=2e6, exchange="NMS")
    ok, reason = s.passes("SMALL", make_df(close=50.0, volume=2e6))
    assert not ok and "mcap" in reason


def test_otc_rejected():
    s = UniverseScreener()
    _inject(s, "OTCX", market_cap=5e9, avg_volume=2e6, exchange="PNK")
    ok, reason = s.passes("OTCX", make_df(close=50.0, volume=2e6))
    assert not ok and "OTC" in reason


def test_qualifying_symbol_passes():
    s = UniverseScreener(ScreenCriteria(min_price=15, min_market_cap=3e9, min_avg_volume=1e6))
    _inject(s, "BIG", market_cap=50e9, avg_volume=8e6, exchange="NMS")
    ok, reason = s.passes("BIG", make_df(close=120.0, volume=8e6))
    assert ok and reason == "ok"
