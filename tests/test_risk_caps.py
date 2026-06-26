"""Tests for the self-managing portfolio caps: the leverage gate and the
single-sector concentration gate on ``PortfolioRisk``.

These gates auto-pause new entries when the book is over-extended and auto-resume
when it recovers — no manual MIN_SCORE override. Defaults of 0 disable a gate.
"""

from __future__ import annotations

from src.risk.portfolio_risk import PortfolioRisk


def _risk(max_leverage=1.5, max_sector_pct=0.35):
    return PortfolioRisk(max_leverage=max_leverage, max_sector_pct=max_sector_pct)


# --------------------------------------------------------------------------- #
# Leverage gate
# --------------------------------------------------------------------------- #
def test_leverage_under_cap_allows_entry():
    r = _risk()
    # gross 100k on 100k equity = 1.0x, adding 20k -> 1.2x, under 1.5x.
    allowed, current, projected, reason = r.leverage_gate(100_000, 100_000, 20_000)
    assert allowed is True
    assert round(current, 2) == 1.0
    assert round(projected, 2) == 1.2
    assert reason == ""


def test_leverage_already_over_cap_pauses_all_entries():
    r = _risk()
    # 160k / 100k = 1.6x, already above 1.5x — paused even with zero new notional.
    allowed, current, _proj, reason = r.leverage_gate(160_000, 100_000, 0.0)
    assert allowed is False
    assert round(current, 2) == 1.6
    assert "at/above" in reason


def test_leverage_entry_that_would_breach_is_blocked():
    r = _risk()
    # 1.4x now, but a 20k add -> 1.6x > 1.5x cap.
    allowed, current, projected, reason = r.leverage_gate(140_000, 100_000, 20_000)
    assert allowed is False
    assert round(current, 2) == 1.4
    assert round(projected, 2) == 1.6
    assert "push leverage" in reason


def test_leverage_entry_exactly_at_cap_is_allowed():
    r = _risk()
    # 1.3x + 20k = exactly 1.5x. Cap is "must not exceed", so == is allowed.
    allowed, _c, projected, _reason = r.leverage_gate(130_000, 100_000, 20_000)
    assert allowed is True
    assert round(projected, 2) == 1.5


def test_leverage_gate_disabled_when_zero():
    r = _risk(max_leverage=0.0)
    allowed, current, projected, reason = r.leverage_gate(500_000, 100_000, 50_000)
    assert allowed is True  # disabled — even 5x passes
    assert (current, projected, reason) == (0.0, 0.0, "")


def test_leverage_gate_safe_on_zero_equity():
    r = _risk()
    allowed, _c, _p, _reason = r.leverage_gate(100_000, 0.0, 10_000)
    assert allowed is True  # no division-by-zero, fails open rather than crashing


# --------------------------------------------------------------------------- #
# Sector concentration gate
# --------------------------------------------------------------------------- #
def test_sector_under_cap_allows_entry():
    r = _risk()
    # Financials at 20k/100k = 20%, +10k -> 30%, under 35%.
    allowed, current, reason = r.sector_gate("Financials", 20_000, 100_000, 10_000)
    assert allowed is True
    assert round(current, 2) == 0.20
    assert reason == ""


def test_sector_entry_that_would_breach_is_blocked():
    r = _risk()
    # Financials at 33%, +5k -> 38% > 35% cap (the scenario that let it hit 90%).
    allowed, current, reason = r.sector_gate("Financials", 33_000, 100_000, 5_000)
    assert allowed is False
    assert round(current, 2) == 0.33
    assert "Financials" in reason and "cap" in reason


def test_sector_already_over_cap_is_blocked():
    r = _risk()
    allowed, current, reason = r.sector_gate("Financials", 90_000, 100_000, 0.0)
    assert allowed is False
    assert round(current, 2) == 0.90
    assert "skipping new Financials entries" in reason


def test_sector_gate_disabled_when_zero():
    r = _risk(max_sector_pct=0.0)
    allowed, current, reason = r.sector_gate("Financials", 90_000, 100_000, 10_000)
    assert allowed is True
    assert (current, reason) == (0.0, "")


def test_sector_gate_safe_on_missing_sector_or_zero_equity():
    r = _risk()
    assert r.sector_gate("", 50_000, 100_000, 10_000)[0] is True   # no sector label
    assert r.sector_gate("Tech", 50_000, 0.0, 10_000)[0] is True    # zero equity


# --------------------------------------------------------------------------- #
# Defaults: gates are opt-in (off) unless wired from settings
# --------------------------------------------------------------------------- #
def test_caps_default_off():
    r = PortfolioRisk()
    assert r.max_leverage == 0.0 and r.max_sector_pct == 0.0
    assert r.leverage_gate(999_000, 100_000, 100_000)[0] is True
    assert r.sector_gate("Financials", 999_000, 100_000, 100_000)[0] is True
