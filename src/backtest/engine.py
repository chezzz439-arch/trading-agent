"""Event-driven backtester for the EMA/RSI + 5:1 RR strategy.

Uses yfinance for extended historical data, then replays the exact production
pipeline bar-by-bar (strategy -> RR filter -> position sizer), simulating
bracket exits intrabar. Reports total return, Sharpe, max drawdown, win rate,
average realised R-multiple and expectancy, and can plot the equity curve.

Simplifying assumptions (documented so results aren't over-trusted):
* one position per symbol at a time;
* entry at the signal bar's close, exits checked against each later bar's
  high/low with the stop evaluated *before* the target (conservative);
* no commissions or slippage.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.data.feed import is_crypto
from src.signals.rr_filter import RRFilter
from src.signals.strategy import EMAStrategy

logger = logging.getLogger(__name__)

# Approximate bars per year, for annualising the Sharpe ratio.
_PERIODS_PER_YEAR = {"1d": 252, "1h": 252 * 7, "1wk": 52, "4h": 252 * 2}


@dataclass
class BacktestResult:
    symbol: str
    total_return: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    avg_rr: float
    expectancy: float
    num_trades: int
    final_equity: float
    equity_curve: pd.Series = field(repr=False)
    trades: pd.DataFrame = field(repr=False)

    def summary(self) -> str:
        return (
            f"{self.symbol}: trades={self.num_trades} "
            f"return={self.total_return * 100:.1f}% sharpe={self.sharpe:.2f} "
            f"maxDD={self.max_drawdown * 100:.1f}% win={self.win_rate * 100:.1f}% "
            f"avgR={self.avg_rr:.2f} expectancy={self.expectancy:.2f}R "
            f"final=${self.final_equity:,.0f}"
        )


class Backtester:
    def __init__(
        self,
        strategy: Optional[EMAStrategy] = None,
        rr_filter: Optional[RRFilter] = None,
        initial_capital: float = 100_000.0,
        risk_per_trade: float = 0.01,
    ) -> None:
        self.strategy = strategy or EMAStrategy()
        self.rr_filter = rr_filter or RRFilter()
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade

    # ------------------------------------------------------------------ #
    # Data loading (yfinance)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _yf_symbol(symbol: str) -> str:
        # BTC/USD -> BTC-USD for Yahoo Finance.
        return symbol.replace("/", "-") if is_crypto(symbol) else symbol

    def load_data(self, symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        import yfinance as yf

        ticker = self._yf_symbol(symbol)
        df = yf.download(ticker, period=period, interval=interval,
                         auto_adjust=True, progress=False)
        if df is None or df.empty:
            logger.warning("yfinance returned no data for %s", ticker)
            return pd.DataFrame()
        # Newer yfinance returns a column MultiIndex even for a single ticker.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=str.lower)
        keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
        return df[keep].dropna()

    # ------------------------------------------------------------------ #
    # Simulation
    # ------------------------------------------------------------------ #
    def run(self, symbol: str, period: str = "2y", interval: str = "1d") -> Optional[BacktestResult]:
        df = self.load_data(symbol, period, interval)
        if df.empty or len(df) < self.strategy.min_bars + 5:
            logger.warning("Not enough data to backtest %s", symbol)
            return None

        fractional = is_crypto(symbol)
        equity = self.initial_capital
        warmup = self.strategy.min_bars
        position = None  # dict: side, entry, stop, target, qty, risk_per_share
        trades: list[dict] = []
        curve_idx, curve_val = [], []

        for i in range(warmup, len(df)):
            window = df.iloc[: i + 1]
            bar = df.iloc[i]

            if position is not None:
                exit_price, outcome = self._check_exit(position, bar)
                if exit_price is not None:
                    pnl = self._pnl(position, exit_price)
                    equity += pnl
                    r_multiple = (pnl / position["qty"]) / position["risk_per_share"]
                    trades.append({
                        "entry_time": position["entry_time"],
                        "exit_time": df.index[i],
                        "side": position["side"],
                        "entry": position["entry"],
                        "exit": exit_price,
                        "qty": position["qty"],
                        "pnl": pnl,
                        "r_multiple": r_multiple,
                        "outcome": outcome,
                    })
                    position = None

            if position is None:
                position = self._maybe_enter(symbol, window, equity, fractional, df.index[i])

            # Mark-to-market equity for the curve.
            mtm = equity + (self._pnl(position, bar["close"]) if position else 0.0)
            curve_idx.append(df.index[i])
            curve_val.append(mtm)

        equity_curve = pd.Series(curve_val, index=curve_idx, name="equity")
        return self._metrics(symbol, equity_curve, trades, interval)

    def _maybe_enter(self, symbol, window, equity, fractional, ts):
        from src.risk.position_sizer import PositionSizer

        signal = self.strategy.evaluate(symbol, window)
        if signal is None:
            return None
        plan = self.rr_filter.evaluate(signal, window)
        if plan is None:
            return None
        sized = PositionSizer(self.risk_per_trade).size(plan, equity, fractional=fractional)
        if sized is None:
            return None
        return {
            "side": plan.side,
            "entry": plan.entry,
            "stop": plan.stop,
            "target": plan.target,
            "qty": sized.qty,
            "risk_per_share": plan.risk_per_share,
            "entry_time": ts,
        }

    @staticmethod
    def _check_exit(position, bar):
        """Return (exit_price, outcome) if stop/target hit this bar, else (None, None)."""
        if position["side"] == "long":
            if bar["low"] <= position["stop"]:
                return position["stop"], "loss"
            if bar["high"] >= position["target"]:
                return position["target"], "win"
        else:  # short
            if bar["high"] >= position["stop"]:
                return position["stop"], "loss"
            if bar["low"] <= position["target"]:
                return position["target"], "win"
        return None, None

    @staticmethod
    def _pnl(position, exit_price) -> float:
        direction = 1 if position["side"] == "long" else -1
        return direction * (exit_price - position["entry"]) * position["qty"]

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #
    def _metrics(self, symbol, equity_curve, trades, interval) -> BacktestResult:
        trades_df = pd.DataFrame(trades)
        final_equity = float(equity_curve.iloc[-1]) if len(equity_curve) else self.initial_capital
        total_return = final_equity / self.initial_capital - 1.0

        # Sharpe from per-bar returns of the equity curve.
        rets = equity_curve.pct_change().dropna()
        ppy = _PERIODS_PER_YEAR.get(interval, 252)
        if len(rets) > 1 and rets.std() > 0:
            sharpe = float(rets.mean() / rets.std() * math.sqrt(ppy))
        else:
            sharpe = 0.0

        # Max drawdown.
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        max_dd = float(drawdown.min()) if len(drawdown) else 0.0

        if not trades_df.empty:
            wins = (trades_df["outcome"] == "win").sum()
            win_rate = wins / len(trades_df)
            avg_rr = float(trades_df["r_multiple"].mean())
            expectancy = float(trades_df["r_multiple"].mean())  # mean R per trade
        else:
            win_rate = avg_rr = expectancy = 0.0

        return BacktestResult(
            symbol=symbol,
            total_return=total_return,
            sharpe=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            avg_rr=avg_rr,
            expectancy=expectancy,
            num_trades=len(trades_df),
            final_equity=final_equity,
            equity_curve=equity_curve,
            trades=trades_df,
        )

    # ------------------------------------------------------------------ #
    # Plotting
    # ------------------------------------------------------------------ #
    @staticmethod
    def plot_equity(result: BacktestResult, out_path: str | None = None) -> str:
        """Plot the equity curve to a PNG and return the file path."""
        import matplotlib
        matplotlib.use("Agg")  # headless / no display required
        import matplotlib.pyplot as plt

        out_path = out_path or f"logs/equity_{result.symbol.replace('/', '-')}.png"
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(result.equity_curve.index, result.equity_curve.values, lw=1.3)
        ax.set_title(f"Equity Curve — {result.symbol} ({result.summary()})", fontsize=9)
        ax.set_xlabel("Time")
        ax.set_ylabel("Equity ($)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        logger.info("Saved equity curve to %s", out_path)
        return out_path
