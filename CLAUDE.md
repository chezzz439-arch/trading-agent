# Trading Agent — Project Guide

A production-style algorithmic trading agent for Alpaca (paper by default). It
trades an EMA(20/50) crossover with RSI confirmation, gated by a strict **5:1
reward-to-risk filter**, multi-timeframe trend alignment, and portfolio-level
risk controls. Orders are submitted as atomic bracket orders.

## Pipeline

For each watchlist symbol, every scan:

```
multi-timeframe alignment (1h & 4h agree)
  -> EMA/RSI signal on the entry timeframe (must match the aligned direction)
  -> 5:1 RR filter (ATR stop + swing-structure target; reject if RR < 5)
  -> portfolio gates (max positions, correlation, daily-loss kill switch)
  -> position sizing (1% account risk, 10% max position)
  -> bracket order (entry + stop-loss + take-profit)
```

## Project structure

```
config/
  settings.py            All tunable parameters + .env credential loading
src/
  data/feed.py           MarketFeed — Alpaca stock + crypto OHLCV bars
  signals/
    strategy.py          EMAStrategy — EMA crossover + RSI; .evaluate() / .bias()
    rr_filter.py         RRFilter — ATR stops, swing targets, 5:1 gate; atr()
    multi_timeframe.py   MultiTimeframe — 1h/4h directional agreement
  risk/
    position_sizer.py    PositionSizer — fixed-fractional sizing + position cap
    portfolio_risk.py    PortfolioRisk — kill switch, max positions, correlation
  execution/broker.py    Broker — Alpaca TradingClient + bracket orders
  backtest/engine.py     Backtester — yfinance replay, metrics, equity plot
main.py                  Live loop, logging, graceful shutdown
tests/test_rr_filter.py  Unit tests (RR filter, ATR, sizing)
```

## Setup

1. Create `.env` in the project root (it is git-ignored):
   ```
   ALPACA_API_KEY=...
   ALPACA_SECRET_KEY=...
   ALPACA_BASE_URL=https://paper-api.alpaca.markets
   ```
2. `source venv/bin/activate`
3. `pip install -r requirements.txt`

## Running

```bash
python main.py                 # live paper loop (Ctrl+C for graceful shutdown)
python -m pytest tests/        # unit tests
python -c "from src.backtest.engine import Backtester; \
  r = Backtester().run('AAPL', period='2y', interval='1d'); \
  print(r.summary()); Backtester.plot_equity(r)"
```

Logs are written to `logs/agent_YYYYMMDD.log`.

## How to adjust risk parameters

All risk/strategy knobs live in `config/settings.py` — change them there, not in
the modules:

| Parameter | Meaning | Default |
|-----------|---------|---------|
| `RISK_PER_TRADE` | account fraction risked per trade | `0.01` (1%) |
| `MAX_POSITION_PCT` | single-position notional cap | `0.10` (10%) |
| `RR_RATIO` | minimum reward:risk to take a trade | `5.0` |
| `ATR_MULTIPLIER` | stop distance = ATR × this | `1.5` |
| `MAX_CONCURRENT_POSITIONS` | open-position cap | `3` |
| `DAILY_LOSS_LIMIT` | kill-switch drawdown from day start | `0.03` (3%) |
| `MAX_CORRELATION` | block new position if corr ≥ this | `0.80` |
| `SCAN_INTERVAL` | seconds between scans | `300` |

## How to add a new strategy

1. Create `src/signals/<your_strategy>.py` with a class exposing:
   - `evaluate(symbol, df) -> Signal | None` — fires only on a fresh entry trigger;
   - `bias(df) -> "long" | "short" | None` — standing direction (for MTF alignment).
   Return the shared `Signal` dataclass from `src/signals/strategy.py` so the RR
   filter, sizer, and broker work unchanged.
2. Indicators: reuse `ema`/`rsi` in `strategy.py` and `atr` in `rr_filter.py`, or
   add new helpers next to them.
3. Wire it in `main.py::TradingAgent.__init__` (swap `EMAStrategy`) and, if it
   needs new knobs, add them to `config/settings.py`.
4. Add unit tests under `tests/` following `test_rr_filter.py`.

## Conventions & gotchas

- **Type hints + docstrings** on all public classes/methods.
- **Never crash on one bad bar/API call**: data and signal paths catch
  exceptions, log, and return empty/`None` so a single failure skips the symbol
  rather than killing the loop.
- **Crypto limitations**: Alpaca crypto does **not** support bracket orders or
  shorting. `Broker` degrades to a simple market entry for crypto longs (logging
  a warning) and refuses crypto shorts. Manage crypto stops/targets separately
  or extend `Broker` with OCO logic.
- **Data feed**: stocks use the free `iex` feed (15-min delayed) by default; set
  `ALPACA_DATA_FEED=sip` if you have a paid subscription.
- **Backtester** assumes one position per symbol, entry at signal-bar close, and
  no commission/slippage — treat metrics as indicative, not exact.
```
