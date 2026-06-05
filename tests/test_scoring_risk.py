"""Tests for the master scorer, portfolio risk gates, technical patterns,
and the monitoring dashboard."""

from __future__ import annotations

import pandas as pd
import pytest

from src.monitoring.dashboard import Dashboard, DashboardState
from src.risk.portfolio_risk import PortfolioRisk
from src.signals.ml_signals import MLPrediction
from src.signals.mtf import MTFResult, TimeframeView
from src.signals.quant import QuantResult
from src.signals.regime import Regime
from src.signals.rr_filter import TradePlan
from src.signals.scorer import MasterScorer
from src.signals.technical import TechnicalAnalysis, candle_patterns


# --------------------------------------------------------------------------- #
# Master scorer
# --------------------------------------------------------------------------- #
def _strong_components():
    from src.signals.technical import TechnicalResult
    tech = TechnicalResult(
        values={"stoch_k": 60.0, "rsi14": 60.0},
        signals={"above_key_mas": 4, "bullish_pattern": True, "bearish_pattern": False,
                 "volume_confirms": True, "rsi_bull": True, "rsi_overbought": False,
                 "macd_bull": True},
    )
    quant = QuantResult(values={"hurst": 0.62, "zscore_20": 0.5, "slope_pct": 1.2,
                                "autocorr_lag1": 0.1})
    regime = Regime(volatility="medium", trend="strong_trend", strategy="momentum")
    mtf = MTFResult("X", views={"1Day": TimeframeView("1Day", "long", "bullish")},
                    dominant_direction="long", confluence_score=5)
    ml = MLPrediction(direction="up", agreement=True, confidence=0.8, trained=True)
    plan = TradePlan("X", "long", 100, 97, 115, rr=5.0, risk_per_share=3, atr=2, reason="")
    return tech, quant, regime, mtf, ml, plan


def test_strong_setup_scores_high_and_passes():
    tech, quant, regime, mtf, ml, plan = _strong_components()
    s = MasterScorer().score("X", "long", technical=tech, quant=quant, regime=regime,
                             mtf=mtf, ml=ml, plan=plan)
    assert s.total >= 70
    assert s.passed
    # Every dimension should be at or near its cap for an ideal long.
    assert s.breakdown["technical"] == pytest.approx(20.0)
    assert s.breakdown["momentum"] == pytest.approx(15.0)
    assert s.breakdown["mtf"] == pytest.approx(15.0)


def test_prerank_score_is_technical_plus_momentum():
    tech, *_ = _strong_components()
    pre = MasterScorer().prerank_score("X", "long", tech)
    # Strong long setup: technical (20) + momentum (15) caps at 35.
    assert pre == pytest.approx(35.0)
    # Bounded to the 0-35 range for any input.
    assert 0 <= MasterScorer().prerank_score("X", "short", tech) <= 35


def test_empty_components_do_not_pass():
    # No analysis at all -> neutral fallbacks, no RR -> below the 70 gate.
    s = MasterScorer().score("X", "long")
    assert not s.passed
    assert s.total < 70


def test_rr_target_alignment_gives_full_marks():
    # A 4:1 plan should earn full RR marks (8.0) when the scorer's target is 4:1,
    # but only partial (4.8) when the target stays at 5:1.
    plan4 = TradePlan("X", "long", 100, 98, 108, rr=4.0, risk_per_share=2, atr=2, reason="")
    dummy = MasterScorer().score("X", "long")  # throwaway TradeScore for the helper
    assert MasterScorer(rr_target=5.0)._risk_reward(plan4, dummy) == pytest.approx(4.8)
    assert MasterScorer(rr_target=4.0)._risk_reward(plan4, dummy) == pytest.approx(8.0)
    # 5:1 still earns full marks at the 5:1 target (unchanged default behaviour).
    plan5 = TradePlan("X", "long", 100, 98, 110, rr=5.0, risk_per_share=2, atr=2, reason="")
    assert MasterScorer(rr_target=5.0)._risk_reward(plan5, dummy) == pytest.approx(8.0)


def test_mtf_opposed_direction_scores_zero():
    tech, quant, regime, mtf, ml, plan = _strong_components()
    mtf.dominant_direction = "short"   # opposes the long candidate
    s = MasterScorer().score("X", "long", technical=tech, quant=quant, regime=regime,
                             mtf=mtf, ml=ml, plan=plan)
    assert s.breakdown["mtf"] == 0.0


# --------------------------------------------------------------------------- #
# Portfolio risk
# --------------------------------------------------------------------------- #
def test_risk_fraction_scales_with_score():
    r = PortfolioRisk()
    assert r.risk_fraction_for_score(90) == 0.02
    assert r.risk_fraction_for_score(80) == 0.01
    assert r.risk_fraction_for_score(72) == 0.005
    assert r.risk_fraction_for_score(60) == 0.0


def test_kill_switch_daily_weekly_and_streak():
    r = PortfolioRisk(daily_loss_limit=0.03, weekly_loss_limit=0.07, max_consecutive_losses=5)
    r.set_day_start_equity(100_000)
    assert not r.kill_switch_triggered(98_000)        # -2% daily, ok
    assert r.kill_switch_triggered(96_900)            # -3.1% daily -> trip

    r2 = PortfolioRisk()
    r2.set_day_start_equity(100_000)
    r2.set_week_start_equity(100_000)
    assert r2.kill_switch_triggered(92_000)           # -8% weekly -> trip

    r3 = PortfolioRisk(max_consecutive_losses=5)
    r3.set_day_start_equity(100_000)
    for _ in range(5):
        r3.record_trade_result(-100)
    assert r3.kill_switch_triggered(100_000)          # 5 straight losses -> trip


def test_portfolio_heat_cap():
    r = PortfolioRisk(portfolio_heat_max=0.06)
    assert r.heat_allows([2_000, 2_000], 2_000, 100_000)      # exactly 6%
    assert not r.heat_allows([2_000, 2_000], 2_100, 100_000)  # over 6%


def test_pre_trade_check_blocks_low_score_rr_and_corr():
    r = PortfolioRisk(min_score=70, min_rr=5.0, max_correlation=0.70)
    r.set_day_start_equity(100_000)
    assert not r.pre_trade_check(60, 6.0, 0, current_equity=100_000).allowed   # score
    assert not r.pre_trade_check(80, 4.0, 0, current_equity=100_000).allowed   # rr
    assert not r.pre_trade_check(80, 6.0, 0, candidate_corr=0.85,
                                 current_equity=100_000).allowed               # corr
    ok = r.pre_trade_check(80, 6.0, 0, candidate_corr=0.2, current_equity=100_000)
    assert ok.allowed and ok.risk_fraction == 0.01


def test_manage_stop_breakeven_and_trail():
    r = PortfolioRisk()
    # Long, entry 100, risk/share 2. At +2R move stop to breakeven (entry).
    assert r.manage_stop("long", 100, 104, 98, 2) == 100
    # At +3R trail by 1 ATR below price.
    assert r.manage_stop("long", 100, 106, 98, 2, atr=1) == 105
    # Never loosens an already-better stop.
    assert r.manage_stop("long", 100, 104, 101, 2) == 101


# --------------------------------------------------------------------------- #
# Technical patterns
# --------------------------------------------------------------------------- #
def test_bullish_engulfing_detected():
    # bar -2 bearish (10->9), bar -1 bullish engulfs it (8.5->10.5).
    rows = [(10, 11, 9, 10)] * 3 + [(10, 10.2, 8.9, 9.0), (8.5, 10.6, 8.4, 10.5)]
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    pats = candle_patterns(df)
    assert pats["bullish_engulfing"]


# --------------------------------------------------------------------------- #
# Dashboard
# --------------------------------------------------------------------------- #
def test_dashboard_render_and_report(tmp_path):
    st = DashboardState(
        equity=101_000, buying_power=400_000, daily_pnl=1_000, weekly_pnl=1_000,
        regime_label="medium_vol/strong_trend/momentum", risk_state="risk_on",
        open_positions=[{"symbol": "AAPL", "qty": 50, "pnl": 250.0, "pnl_pct": 1.2}],
        scores=[{"symbol": "NVDA", "side": "long", "score": 82.0, "passed": True}],
        closed_today=[{"symbol": "TSLA", "pnl": -120.0, "r_multiple": -1.0}],
        win_rate_20=0.55,
    )
    out = Dashboard().render(st)
    assert "TRADING AGENT" in out and "AAPL" in out and "NVDA" in out

    d = Dashboard(log_dir=str(tmp_path))
    path = d.daily_report(st)
    assert path.endswith(".txt")
    assert "DAILY REPORT" in open(path).read()
