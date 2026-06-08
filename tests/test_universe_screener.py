"""Tests for the universe screener's sector exclusion."""

from __future__ import annotations

import scripts.universe_screener as us


def test_excluded_sectors_includes_financials():
    assert "financials" in us.EXCLUDE_SECTORS


def test_screen_drops_financials(monkeypatch):
    """screen_stocks must remove Financials before any data fetch."""
    uni = {
        "AAPL": {"name": "Apple", "sector": "Technology"},
        "JPM": {"name": "JPMorgan", "sector": "Financials"},
        "MS": {"name": "Morgan Stanley", "sector": "Financials"},
        "XOM": {"name": "Exxon", "sector": "Energy"},
    }
    captured = {}

    # Stub the network passes: capture which symbols survive the sector filter.
    def fake_download(part, **kwargs):
        captured["downloaded"] = set(part)
        raise RuntimeError("stop after capture")  # bail before yfinance parsing

    import types
    fake_yf = types.SimpleNamespace(download=fake_download)
    monkeypatch.setitem(__import__("sys").modules, "yfinance", fake_yf)

    us.screen_stocks(uni)

    downloaded = captured["downloaded"]
    assert "JPM" not in downloaded and "MS" not in downloaded
    assert "AAPL" in downloaded and "XOM" in downloaded


def test_sector_match_is_case_insensitive(monkeypatch):
    uni = {"GS": {"name": "Goldman", "sector": "FINANCIALS"}}
    captured = {"downloaded": set()}

    def fake_download(part, **kwargs):
        captured["downloaded"] = set(part)
        raise RuntimeError("stop")

    import types
    monkeypatch.setitem(__import__("sys").modules, "yfinance",
                        types.SimpleNamespace(download=fake_download))
    qualified, funnel = us.screen_stocks(uni)
    # GS filtered out -> nothing downloaded, nothing qualifies
    assert captured["downloaded"] == set()
    assert qualified == []
    assert funnel["sector_excluded"] == 1
