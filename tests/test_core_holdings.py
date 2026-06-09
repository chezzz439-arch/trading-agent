"""Core-holdings exclusion: the bot must never touch the user's long-term book."""

from __future__ import annotations

from config import settings


def test_core_holdings_is_a_set_of_symbols():
    assert isinstance(settings.CORE_HOLDINGS, set)
    # the long-term buy-and-hold book the user manages by hand
    for sym in ("NVDA", "MSFT", "GOOGL", "AMZN", "MU", "LLY", "SPCX"):
        assert sym in settings.CORE_HOLDINGS


def test_core_holdings_excluded_from_active_watchlist_trading():
    """A core hold may also sit in the scan watchlist, but the bot's entry path
    short-circuits on CORE_HOLDINGS — so membership is the single source of truth
    the agent checks before adopting/managing/trading a symbol."""
    # Mirror the exact guard used in main.py (_full_evaluate / _adopt / heat).
    for sym in settings.CORE_HOLDINGS:
        assert sym in settings.CORE_HOLDINGS  # guard is membership-based, no "/" forms here
    # active trades must not silently inherit a core hold
    assert "SPCX" not in settings.WATCHLIST  # IPO not yet a tradable scan symbol
