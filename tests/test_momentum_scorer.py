"""Tests for the pure-momentum-long-only scorer refactor.

Locks the two hard gates (no shorts; must outperform SPY) and confirms the
mean-reversion scoring components are gone.
"""

from __future__ import annotations

from src.signals.scorer import MAX_POINTS, MasterScorer
from src.signals.quant import QuantResult


def _quant(rs20, rs60, hurst=0.6, slope=1.0, ac=0.1):
    outperf = (rs20 > 0 and rs60 > 0) if (rs20 is not None and rs60 is not None) else None
    return QuantResult(values={
        "rel_strength_20": rs20, "rel_strength_60": rs60, "rs_outperform": outperf,
        "hurst": hurst, "slope_pct": slope, "autocorr_lag1": ac, "zscore_20": -2.0,
    })


def test_weights_sum_to_100_with_relative_strength():
    assert sum(MAX_POINTS.values()) == 100
    assert "relative_strength" in MAX_POINTS


def test_shorts_are_vetoed():
    sc = MasterScorer(min_score=70).score("X", "short", quant=_quant(0.05, 0.05))
    assert sc.rs_veto is True
    assert sc.passed is False


def test_long_lagging_spy_is_vetoed_even_if_high_score():
    # underperforms SPY on the 20-day horizon -> hard gate vetoes regardless of score
    sc = MasterScorer(min_score=70).score("X", "long", quant=_quant(-0.02, 0.01))
    assert sc.rs_veto is True
    assert sc.passed is False


def test_long_outperforming_spy_not_vetoed():
    sc = MasterScorer(min_score=70).score("X", "long", quant=_quant(0.05, 0.08))
    assert sc.rs_veto is False
    assert sc.breakdown["relative_strength"] > 0


def test_no_spy_series_is_neutral_not_vetoed():
    # backtest/live without a market series -> neutral RS, never a veto
    sc = MasterScorer(min_score=70).score("X", "long", quant=_quant(None, None))
    assert sc.rs_veto is False
    assert sc.breakdown["relative_strength"] == MAX_POINTS["relative_strength"] * 0.5


def test_statistical_is_pure_trend_no_mean_reversion():
    # A strongly negative z-score (an oversold/mean-reversion setup) must NOT add
    # points now; only trend persistence (hurst/slope/autocorr) is rewarded.
    trend = MasterScorer(min_score=70).score("X", "long", quant=_quant(0.05, 0.05, hurst=0.6, slope=1.0, ac=0.1))
    flat = MasterScorer(min_score=70).score("X", "long", quant=_quant(0.05, 0.05, hurst=0.5, slope=-1.0, ac=-0.1))
    assert trend.breakdown["statistical"] > flat.breakdown["statistical"]
    assert trend.breakdown["statistical"] <= MAX_POINTS["statistical"]
