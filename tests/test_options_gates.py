"""Unit tests for the rebuilt options entry-quality gates (2026-06-23).

Covers every gate the live path depends on, as pure functions (no network):
post-open delay, spread, Greeks selection, theta, IV-rank (true + HV proxy),
limit pricing, fill-sanity (phantom-fill guard) and the IV history store.
"""

from __future__ import annotations

import math
from datetime import date, datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo

from src.signals.options_strategy import (
    IVHistoryStore,
    OptionQuote,
    fill_price_is_sane,
    iv_rank,
    iv_rank_proxy_from_hv,
    limit_entry_price,
    minutes_since_open,
    passes_spread,
    realized_vol_series,
    resolve_iv_rank,
    select_by_greeks,
    spread_pct,
    theta_decay_pct,
    within_open_delay,
)

ET = ZoneInfo("America/New_York")


def _q(strike, premium, *, bid, ask, delta, theta=-0.02, vega=0.05, iv=0.30,
       exp_days=37, typ="call"):
    exp = (date.today() + timedelta(days=exp_days)).isoformat()
    return OptionQuote(symbol=f"X{strike}{typ}", underlying="AAPL", type=typ,
                       strike=strike, expiration=exp, premium=premium, bid=bid,
                       ask=ask, delta=delta, theta=theta, vega=vega, iv=iv)


# --- Gate 1: post-open entry delay ----------------------------------------- #
def test_minutes_since_open_basic():
    now = datetime(2026, 6, 23, 9, 45, tzinfo=ET)   # 15 min after 9:30
    assert minutes_since_open(now) == 15.0


def test_within_open_delay_blocks_at_645_pt_equiv():
    # 9:45 ET == 6:45 PT, 15 min after open → inside a 45-min window → BLOCKED.
    now = datetime(2026, 6, 23, 9, 45, tzinfo=ET)
    assert within_open_delay(now, 45) is True


def test_within_open_delay_allows_after_window():
    # 10:30 ET == 7:30 PT, 60 min after open → past the 45-min window → ALLOWED.
    now = datetime(2026, 6, 23, 10, 30, tzinfo=ET)
    assert within_open_delay(now, 45) is False


def test_within_open_delay_false_before_open():
    now = datetime(2026, 6, 23, 9, 0, tzinfo=ET)    # pre-open
    assert within_open_delay(now, 45) is False


def test_within_open_delay_boundary_exact():
    now = datetime(2026, 6, 23, 9, 30, tzinfo=ET)   # exactly at open
    assert within_open_delay(now, 45) is True
    edge = datetime(2026, 6, 23, 10, 15, tzinfo=ET)  # exactly 45 min → allowed
    assert within_open_delay(edge, 45) is False


def test_minutes_since_open_converts_non_et_zone():
    # A true-ET-09:45 instant carried as its UTC value must still read 15 min.
    utc = ZoneInfo("UTC")
    inst = datetime(2026, 6, 23, 9, 45, tzinfo=ET).astimezone(utc)
    assert abs(minutes_since_open(inst) - 15.0) < 1e-6
    # and the gate must still BLOCK it (the dangerous bypass the audit found).
    assert within_open_delay(inst, 45) is True


def test_minutes_since_open_pacific_midday_not_blocked():
    # 12:40 ET (past window) carried in US/Pacific must NOT be wrongly blocked.
    pac = ZoneInfo("America/Los_Angeles")
    inst = datetime(2026, 6, 23, 12, 40, tzinfo=ET).astimezone(pac)
    assert within_open_delay(inst, 45) is False


def test_within_open_delay_negative_config_no_crash():
    now = datetime(2026, 6, 23, 9, 45, tzinfo=ET)
    assert within_open_delay(now, -10) is False   # negative clamps to 0 → off


# --- Gate 3: spread --------------------------------------------------------- #
def test_spread_pct_normal():
    assert abs(spread_pct(1.00, 1.10) - (0.10 / 1.05)) < 1e-9


def test_spread_pct_rejects_one_sided():
    assert spread_pct(0, 1.10) is None
    assert spread_pct(1.0, 0) is None
    assert spread_pct(1.2, 1.0) is None   # crossed


def test_passes_spread_tight_vs_wide():
    tight = _q(200, 1.0, bid=0.98, ask=1.02, delta=0.6)   # ~4%
    wide = _q(200, 1.0, bid=0.90, ask=1.10, delta=0.6)    # ~20%
    assert passes_spread(tight, 0.15) is True
    assert passes_spread(wide, 0.15) is False


# --- Gate 4: theta ---------------------------------------------------------- #
def test_theta_decay_pct():
    assert abs(theta_decay_pct(-0.05, 2.50) - 0.02) < 1e-9
    assert theta_decay_pct(None, 2.5) is None
    assert theta_decay_pct(-0.05, 0) is None


# --- Gate 4: Greeks-based selection ----------------------------------------- #
def test_select_by_greeks_picks_in_delta_band():
    quotes = [
        _q(190, 5.0, bid=4.95, ask=5.05, delta=0.85),   # too deep ITM
        _q(200, 3.0, bid=2.97, ask=3.03, delta=0.62),   # in band ✓ (closest to 0.625 mid)
        _q(205, 2.0, bid=1.98, ask=2.02, delta=0.45),   # too low
    ]
    pick = select_by_greeks(quotes, delta_min=0.55, delta_max=0.70,
                            max_spread_pct=0.15, max_theta_pct=0.05)
    assert pick is not None and pick.strike == 200


def test_select_by_greeks_rejects_wide_spread_even_if_delta_ok():
    quotes = [_q(200, 3.0, bid=2.7, ask=3.3, delta=0.62)]   # 20% spread
    pick = select_by_greeks(quotes, delta_min=0.55, delta_max=0.70,
                            max_spread_pct=0.15, max_theta_pct=0.05)
    assert pick is None


def test_select_by_greeks_rejects_high_theta():
    quotes = [_q(200, 1.0, bid=0.99, ask=1.01, delta=0.62, theta=-0.05)]  # 5%/day
    pick = select_by_greeks(quotes, delta_min=0.55, delta_max=0.70,
                            max_spread_pct=0.15, max_theta_pct=0.02)
    assert pick is None


def test_select_by_greeks_handles_put_negative_delta():
    quotes = [_q(200, 3.0, bid=2.97, ask=3.03, delta=-0.60, typ="put")]
    pick = select_by_greeks(quotes, delta_min=0.55, delta_max=0.70,
                            max_spread_pct=0.15, max_theta_pct=0.05)
    assert pick is not None and pick.type == "put"


def test_select_by_greeks_empty_when_nothing_qualifies():
    quotes = [_q(200, 3.0, bid=2.97, ask=3.03, delta=0.20)]  # delta too low
    assert select_by_greeks(quotes, delta_min=0.55, delta_max=0.70,
                            max_spread_pct=0.15, max_theta_pct=0.05) is None


def test_select_by_greeks_rejects_nan_theta():
    # A NaN theta must NOT slip through the theta gate (NaN > x is False).
    q = _q(200, 1.0, bid=0.99, ask=1.01, delta=0.62, theta=float("nan"))
    assert select_by_greeks([q], delta_min=0.55, delta_max=0.70,
                            max_spread_pct=0.15, max_theta_pct=0.02) is None


def test_select_by_greeks_rejects_nan_premium_and_delta():
    qp = _q(200, float("nan"), bid=0.99, ask=1.01, delta=0.62)
    qd = _q(200, 1.0, bid=0.99, ask=1.01, delta=float("nan"))
    assert select_by_greeks([qp], delta_min=0.55, delta_max=0.70,
                            max_spread_pct=0.15, max_theta_pct=0.05) is None
    assert select_by_greeks([qd], delta_min=0.55, delta_max=0.70,
                            max_spread_pct=0.15, max_theta_pct=0.05) is None


# --- Gate 2: IV rank (true history) ----------------------------------------- #
def test_iv_rank_percentile():
    hist = [0.20, 0.30, 0.40, 0.50, 0.60]   # min .20 max .60
    assert abs(iv_rank(0.20, hist, min_samples=5) - 0.0) < 1e-6
    assert abs(iv_rank(0.60, hist, min_samples=5) - 100.0) < 1e-6
    assert abs(iv_rank(0.40, hist, min_samples=5) - 50.0) < 1e-6


def test_iv_rank_insufficient_history_returns_none():
    assert iv_rank(0.40, [0.3, 0.4], min_samples=20) is None


def test_iv_rank_flat_history_is_indeterminate_none():
    # Flat history is genuinely indeterminate → None so the caller fails CLOSED
    # (a persistently-rich name must not be treated as mid-range and allowed).
    assert iv_rank(0.30, [0.30] * 30, min_samples=20) is None


def test_iv_rank_rejects_nonfinite_current():
    assert iv_rank(float("nan"), [0.2, 0.3, 0.4, 0.5, 0.6], min_samples=5) is None
    assert iv_rank(float("inf"), [0.2, 0.3, 0.4, 0.5, 0.6], min_samples=5) is None


def test_iv_rank_drops_nonfinite_history_then_fails_closed():
    # NaN/inf samples are dropped; if that collapses history to flat → None.
    hist = [0.30] * 25 + [float("nan"), float("inf")]
    assert iv_rank(2.0, hist, min_samples=20) is None


# --- Gate 2: HV bootstrap proxy --------------------------------------------- #
def test_realized_vol_series_constant_is_zero():
    # A flat price series has zero realized vol.
    hv = realized_vol_series([100.0] * 60, window=30)
    assert hv and all(abs(v) < 1e-9 for v in hv)


def test_realized_vol_series_too_short():
    assert realized_vol_series([100, 101, 102], window=30) == []


def test_iv_rank_proxy_ranks_current_iv():
    # Build a price series that oscillates so HV has a real range.
    closes = []
    p = 100.0
    for i in range(400):
        p *= (1.0 + (0.02 if i % 2 == 0 else -0.018))
        closes.append(p)
    r = iv_rank_proxy_from_hv(0.50, closes, window=30)
    assert r is not None and 0.0 <= r <= 100.0


def test_resolve_iv_rank_prefers_history_then_proxy_then_none():
    hist = [0.2, 0.3, 0.4, 0.5, 0.6] * 5   # 25 samples
    r, method = resolve_iv_rank(0.40, hist, [], min_samples=20)
    assert method == "history" and abs(r - 50.0) < 1e-6

    closes = [100.0 * (1.02 if i % 2 == 0 else 0.985) ** 1 for i in range(1, 200)]
    # accumulate properly
    closes = []
    p = 100.0
    for i in range(200):
        p *= (1.02 if i % 2 == 0 else 0.985)
        closes.append(p)
    r2, method2 = resolve_iv_rank(0.40, [], closes, min_samples=20)
    assert method2 == "hv_proxy" and r2 is not None

    r3, method3 = resolve_iv_rank(0.40, [], [], min_samples=20)
    assert method3 == "none" and r3 is None


# --- Gate 3: limit pricing -------------------------------------------------- #
def test_limit_entry_price_mid_and_buffer():
    assert limit_entry_price(1.00, 1.20, 0.0) == 1.10     # exact mid
    assert limit_entry_price(1.00, 1.20, 1.0) == 1.20     # at the ask
    # mid 1.10, +50% of (1.20-1.10)=+0.05 → 1.15
    assert limit_entry_price(1.00, 1.20, 0.5) == 1.15


def test_limit_entry_price_one_sided_none():
    assert limit_entry_price(0, 1.2, 0.02) is None


# --- Gate 5: fill sanity (phantom-fill guard) ------------------------------- #
def test_fill_price_is_sane_accepts_near_market():
    assert fill_price_is_sane(1.05, bid=1.00, ask=1.10) is True


def test_fill_price_is_sane_rejects_zero_none_nan():
    assert fill_price_is_sane(0.0, 1.0, 1.1) is False
    assert fill_price_is_sane(None, 1.0, 1.1) is False
    assert fill_price_is_sane(float("nan"), 1.0, 1.1) is False


def test_fill_price_is_sane_rejects_absurd_print():
    # A 10x-the-ask fill is a garbage print → reject (no phantom position).
    assert fill_price_is_sane(11.0, bid=1.0, ask=1.10) is False


def test_fill_price_is_sane_no_quote_accepts_positive():
    assert fill_price_is_sane(1.5, bid=0, ask=0) is True


# --- IV history store round-trip -------------------------------------------- #
def test_iv_history_store_records_and_caps(tmp_path):
    store = IVHistoryStore(log_dir=str(tmp_path), max_samples=5)
    base = date(2026, 1, 1)
    for i in range(10):
        store.record("AAPL", 0.20 + i * 0.01, day=base + timedelta(days=i))
    hist = store.history("AAPL")
    assert len(hist) == 5                      # capped to max_samples
    assert abs(hist[-1] - 0.29) < 1e-9         # kept the most recent


def test_iv_history_store_keeps_first_clean_sample_per_day(tmp_path):
    store = IVHistoryStore(log_dir=str(tmp_path))
    d = date(2026, 1, 1)
    store.record("AAPL", 0.20, day=d)
    store.record("AAPL", 0.25, day=d)          # same day → first sample kept
    assert store.history("AAPL") == [0.20]


def test_iv_history_store_rejects_garbage(tmp_path):
    store = IVHistoryStore(log_dir=str(tmp_path))
    d = date(2026, 1, 1)
    store.record("AAPL", float("nan"), day=d)
    store.record("AAPL", float("inf"), day=d)
    store.record("AAPL", 0.0, day=d)
    store.record("AAPL", -0.5, day=d)
    store.record("AAPL", 50.0, day=d)          # 5000% IV — out of sane range
    assert store.history("AAPL") == []         # nothing garbage was stored


def test_iv_history_store_persists_across_instances(tmp_path):
    s1 = IVHistoryStore(log_dir=str(tmp_path))
    s1.record("MSFT", 0.33, day=date(2026, 1, 1))
    s2 = IVHistoryStore(log_dir=str(tmp_path))
    assert s2.history("MSFT") == [0.33]
