# Trading Agent — Project Guide

A multi-phase algorithmic trading system for Alpaca (paper by default). It runs
a broad analysis stack — technical, statistical/quant, regime, sentiment,
multi-timeframe, and an ML ensemble — feeds everything into a 0–100 master
score, and only executes trades that clear a 70-point gate plus portfolio-level
risk checks, as 5:1 reward:risk bracket orders.

## Honesty / scope notes (read this first)

- **Deterministic, backtestable core:** technical (P1), quant (P2), regime (P3),
  scorer (P7), RR filter, sizing — these run point-in-time in
  `Backtester.run_pipeline`.
- **Live-only inputs (not point-in-time backtestable):** sentiment (P4, live
  VIX/SPY/sector fetch) and multi-timeframe confluence (P5, needs 5 live
  resolutions). They score *neutral* in the backtest.
- **ML (P6) is honest structure, not validated alpha:** XGBoost + RandomForest
  on technical features, look-ahead-free labels. The **PyTorch LSTM is
  deferred** (`ensemble_ready`/`ML_ENABLED` reflect this). Treat ML output as a
  weak prior, not edge. Validate out-of-sample before trusting it.
- **Crypto:** Alpaca crypto has no bracket orders and no shorting — `Broker`
  degrades to a simple long market entry and refuses crypto shorts.

## The 10-phase pipeline (per symbol, each scan)

```
P1 technical  ─┐
P2 quant      ─┤
P3 regime     ─┼─► P7 master score (0-100) ─► P8 risk gates ─► P9 smart entry ─► P10 dashboard
P4 sentiment  ─┤     (>=70 to trade)            (score, RR,        (bracket /
P5 mtf        ─┤                                 corr, heat,        limit pullback,
P6 ml (XGB+RF)─┘                                 kill switch)       scale-out)
```

Score weights: technical 20, momentum 15, MTF 15, statistical 15, regime 15,
ML 10, risk/reward 10.

## Project structure

```
config/settings.py        All params: 13-symbol watchlist, scores, risk, timeframes
src/
  data/feed.py            MarketFeed — Alpaca stock + crypto OHLCV
  signals/
    technical.py          P1  TechnicalAnalysis — ~60 indicators + derived signals
    quant.py              P2  QuantAnalysis — Hurst/ADF/zscore/beta/MC/cointegration/...
    regime.py             P3  RegimeDetector — vol/trend/market/momentum/volume regimes
    sentiment.py          P4  SentimentAnalyzer — VIX/SPY/sector/dollar/gold/crypto (live)
    mtf.py                P5  MTFConfluence — 15m/1h/4h/1d/1w agreement (live)
    ml_signals.py         P6  MLEnsemble — XGBoost + RandomForest (LSTM TODO)
    scorer.py             P7  MasterScorer — 0-100 fusion, 70 gate
    rr_filter.py              RRFilter — constructed 5:1 target + structural veto
    strategy.py               EMAStrategy + Signal (entry trigger / bias)
    position_sizer.py     -   PositionSizer (used by risk)
  risk/
    portfolio_risk.py     P8  PortfolioRisk — gates, score-sizing, heat, kill switch, stops
    position_sizer.py         fixed-fractional sizing + position cap
  execution/broker.py     P9  Broker — bracket/limit entries, scale-out, dynamic stops
  monitoring/dashboard.py P10 Dashboard — console snapshot, alerts, daily report
  backtest/engine.py          Backtester — run() (EMA) and run_pipeline() (scorer-gated)
main.py                       Live loop wiring all phases
tests/                        test_rr_filter.py, test_scoring_risk.py
```

## Setup & running

```bash
source venv/bin/activate
pip install -r requirements.txt          # incl. ta, scikit-learn, xgboost, statsmodels, scipy
# .env must hold ALPACA_API_KEY / ALPACA_SECRET_KEY / ALPACA_BASE_URL (git-ignored)

python -m pytest tests/                   # unit tests
python -c "from src.backtest.engine import Backtester; \
  print(Backtester().run_pipeline('AAPL', period='2y').summary())"
python main.py                            # live paper loop (Ctrl+C = graceful shutdown + report)
```

## How to tune

Everything lives in `config/settings.py`: `WATCHLIST`, `MIN_SCORE` (70),
`RR_RATIO` (5), `ATR_MULTIPLIER` (1.5), `SWING_LOOKBACK` (100),
`RISK_PER_TRADE`/`MAX_POSITION_PCT`, `MAX_CONCURRENT_POSITIONS`,
`DAILY_LOSS_LIMIT`/`WEEKLY_LOSS_LIMIT`/`MAX_CONSECUTIVE_LOSSES`,
`PORTFOLIO_HEAT_MAX`, `MAX_CORRELATION`, `MTF_TIMEFRAMES`/`MIN_CONFLUENCE`,
`ML_ENABLED`/`ML_RETRAIN_DAYS`.

To change scoring weights, edit `MAX_POINTS` in `scorer.py`. To add an
indicator, add it in `technical.py::TechnicalAnalysis.analyze` and (optionally)
a derived `signals[...]` flag the scorer can read.

## Top upgrades toward institutional quality (cheapest first)

1. Validation harness: commissions + slippage in the backtester, walk-forward
   OOS testing, deflated-Sharpe / Monte-Carlo significance checks.
2. Point-in-time institutional data + execution realism (modeled fills,
   slippage, latency, idempotent order/state reconciliation).
3. Portfolio construction (vol targeting, covariance-aware sizing,
   drawdown/regime-conditional exposure) + a formal research/validation process.
```
