"""Tests for the validation harness: cost model, significance stats, FDR."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.costs import CostModel
from src.backtest.engine import Backtester
from src.backtest.validation import (
    benjamini_hochberg,
    bootstrap_pvalue,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    monte_carlo_sequence,
    probabilistic_sharpe_ratio,
)


# --------------------------------------------------------------------------- #
# Cost model
# --------------------------------------------------------------------------- #
def test_fill_price_is_adverse():
    cm = CostModel(slippage_bps=10)  # 10 bps = 0.1%
    # Long: buy entry fills higher, sell exit fills lower.
    assert cm.fill_price(100.0, "long", is_entry=True) == pytest.approx(100.1)
    assert cm.fill_price(100.0, "long", is_entry=False) == pytest.approx(99.9)
    # Short: sell entry fills lower, buy-to-cover exit fills higher.
    assert cm.fill_price(100.0, "short", is_entry=True) == pytest.approx(99.9)
    assert cm.fill_price(100.0, "short", is_entry=False) == pytest.approx(100.1)


def test_cost_presets_and_zero():
    assert CostModel().is_zero
    assert not CostModel.equities().is_zero
    assert CostModel.crypto().commission_bps > 0


def test_engine_close_applies_costs():
    bt = Backtester(cost_model=CostModel(slippage_bps=10))
    ts = pd.Timestamp("2026-01-01", tz="UTC")
    pos = {"side": "long", "qty": 100, "entry": 100.0, "risk_per_share": 3.0,
           "entry_time": ts, "score": 75}
    pnl, rec = bt._close(pos, ts, 110.0, "win")
    assert pnl < (110 - 100) * 100          # costs reduce the gross $1,000
    assert rec["entry_fill"] > 100 and rec["exit_fill"] < 110


# --------------------------------------------------------------------------- #
# Probabilistic / deflated Sharpe
# --------------------------------------------------------------------------- #
def test_psr_high_for_consistent_gains():
    rng = np.random.default_rng(0)
    r = 0.01 + rng.normal(0, 0.002, 200)   # strongly positive, low vol
    psr = probabilistic_sharpe_ratio(r, 0.0)
    assert psr > 0.99


def test_psr_near_half_for_zero_mean():
    rng = np.random.default_rng(1)
    r = rng.normal(0, 0.01, 300)
    r = r - r.mean()                       # exactly zero sample mean -> no edge
    psr = probabilistic_sharpe_ratio(r, 0.0)
    assert psr == pytest.approx(0.5, abs=1e-6)


def test_deflation_is_stricter_than_psr():
    rng = np.random.default_rng(2)
    r = 0.005 + rng.normal(0, 0.01, 250)
    psr = probabilistic_sharpe_ratio(r, 0.0)
    dsr = deflated_sharpe_ratio(r, num_trials=20, trials_sr_std_perperiod=0.08)
    assert dsr <= psr


def test_expected_max_sharpe_grows_with_trials():
    assert expected_max_sharpe(0.1, 100) > expected_max_sharpe(0.1, 5)


# --------------------------------------------------------------------------- #
# Bootstrap + Monte Carlo
# --------------------------------------------------------------------------- #
def test_bootstrap_pvalue_directions():
    assert bootstrap_pvalue([5.0, 4.0, 6.0, 5.5, 4.5]) < 0.05      # clearly positive
    assert bootstrap_pvalue([-1.0, -1.0, -1.0, -1.0]) > 0.95       # clearly negative


def test_monte_carlo_sequence_keys():
    rng = np.random.default_rng(3)
    pct = rng.normal(0.001, 0.02, 50)
    mc = monte_carlo_sequence(pct, n=500)
    assert {"final_median", "maxdd_p95", "prob_profit"} <= set(mc)
    assert -1.0 <= mc["maxdd_p95"] <= 0.0


# --------------------------------------------------------------------------- #
# Benjamini-Hochberg
# --------------------------------------------------------------------------- #
def test_benjamini_hochberg_rejections():
    # Only the smallest p-value should survive FDR at alpha=0.05.
    res = benjamini_hochberg([0.001, 0.04, 0.5], alpha=0.05)
    assert [r["reject"] for r in res] == [True, False, False]
    assert res[0]["qvalue"] == pytest.approx(0.003, abs=1e-6)


def test_benjamini_hochberg_handles_none():
    res = benjamini_hochberg([None, 0.01, None])
    assert res[0]["qvalue"] is None and res[1]["reject"]
