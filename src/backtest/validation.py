"""Validation harness — turn backtest output into a defensible verdict.

Provides the statistics that separate a real edge from an overfit/lucky one:

* **Probabilistic Sharpe Ratio (PSR)** — confidence the true Sharpe > 0, adjusted
  for sample length, skew and kurtosis (Bailey & Lopez de Prado).
* **Deflated Sharpe Ratio (DSR)** — PSR against the *expected maximum* Sharpe
  from running ``num_trials`` strategies, i.e. correcting for selection bias.
* **Bootstrap p-value** — resample trades to estimate P(mean R-multiple <= 0).
* **Monte-Carlo sequence risk** — resample the trade sequence to get the
  distribution of final return and max drawdown.
* **Benjamini-Hochberg** — FDR correction across the per-symbol p-values.

``Validator`` runs cost-aware pipeline backtests, splits them into walk-forward
out-of-sample folds, and assembles a report with an explicit verdict.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_GAMMA = 0.5772156649015329  # Euler-Mascheroni


# ============================================================================ #
# Significance statistics
# ============================================================================ #
def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _nppf(p: float) -> float:
    from scipy.stats import norm
    return float(norm.ppf(p))


def sharpe_ratio(returns: np.ndarray, periods: int = 252) -> float:
    r = np.asarray(returns, dtype=float)
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * math.sqrt(periods))


def probabilistic_sharpe_ratio(
    returns: np.ndarray, sr_star_annual: float = 0.0, periods: int = 252
) -> Optional[float]:
    """P(true Sharpe > ``sr_star_annual``) given the sample's higher moments."""
    from scipy.stats import kurtosis, skew

    r = np.asarray(returns, dtype=float)
    n = len(r)
    if n < 3 or r.std(ddof=1) == 0:
        return None
    sr = r.mean() / r.std(ddof=1)                  # per-period observed Sharpe
    sr_star = sr_star_annual / math.sqrt(periods)  # benchmark -> per-period
    sk = float(skew(r))
    ku = float(kurtosis(r, fisher=False))          # non-excess kurtosis
    denom = math.sqrt(max(1e-12, 1 - sk * sr + ((ku - 1) / 4) * sr ** 2))
    return float(_ncdf((sr - sr_star) * math.sqrt(n - 1) / denom))


def expected_max_sharpe(trials_sr_std: float, num_trials: int) -> float:
    """Expected maximum *per-period* Sharpe across ``num_trials`` independent trials."""
    n = max(2, int(num_trials))
    a = (1 - _GAMMA) * _nppf(1 - 1.0 / n)
    b = _GAMMA * _nppf(1 - 1.0 / (n * math.e))
    return trials_sr_std * (a + b)


def deflated_sharpe_ratio(
    returns: np.ndarray, num_trials: int, trials_sr_std_perperiod: float,
    periods: int = 252,
) -> Optional[float]:
    """PSR against the expected-max Sharpe of ``num_trials`` (selection-corrected)."""
    sr_star_perperiod = expected_max_sharpe(trials_sr_std_perperiod, num_trials)
    return probabilistic_sharpe_ratio(returns, sr_star_perperiod * math.sqrt(periods), periods)


def bootstrap_pvalue(trade_returns, n_boot: int = 10000, seed: int = 7) -> Optional[float]:
    """One-sided P(mean trade return <= 0) via resampling with replacement."""
    x = np.asarray(trade_returns, dtype=float)
    if len(x) < 2:
        return None
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(np.mean(means <= 0))


def monte_carlo_sequence(trade_returns_pct, n: int = 5000, seed: int = 7) -> dict:
    """Resample the trade sequence to get final-return and max-drawdown distributions.

    ``trade_returns_pct`` are per-trade returns as fractions of equity.
    """
    x = np.asarray(trade_returns_pct, dtype=float)
    if len(x) < 2:
        return {}
    rng = np.random.default_rng(seed)
    finals, maxdds = [], []
    for _ in range(n):
        path = np.cumprod(1 + rng.choice(x, size=len(x), replace=True))
        finals.append(path[-1] - 1)
        peak = np.maximum.accumulate(path)
        maxdds.append(float(((path - peak) / peak).min()))
    finals, maxdds = np.array(finals), np.array(maxdds)
    return {
        "final_median": float(np.median(finals)),
        "final_p05": float(np.percentile(finals, 5)),
        "final_p95": float(np.percentile(finals, 95)),
        "prob_profit": float(np.mean(finals > 0)),
        "maxdd_median": float(np.median(maxdds)),
        "maxdd_p95": float(np.percentile(maxdds, 5)),  # 5th pct = worst-tail DD
    }


def benjamini_hochberg(pvalues: list[float], alpha: float = 0.05) -> list[dict]:
    """FDR correction. Returns per-input {p, qvalue, reject} preserving input order."""
    valid = [(i, p) for i, p in enumerate(pvalues) if p is not None and np.isfinite(p)]
    m = len(valid)
    out = [{"p": p, "qvalue": None, "reject": False} for p in pvalues]
    if m == 0:
        return out
    ordered = sorted(valid, key=lambda t: t[1])
    qvals = {}
    prev_q = 1.0
    for rank in range(m, 0, -1):
        idx, p = ordered[rank - 1]
        q = min(prev_q, p * m / rank)
        qvals[idx] = q
        prev_q = q
    for idx, q in qvals.items():
        out[idx]["qvalue"] = float(q)
        out[idx]["reject"] = bool(q <= alpha)
    return out


# ============================================================================ #
# Validator
# ============================================================================ #
@dataclass
class SymbolValidation:
    symbol: str
    num_trades: int
    total_return: float
    sharpe: float
    psr: Optional[float]
    bootstrap_p: Optional[float]
    folds: list = field(default_factory=list)     # per-fold OOS summaries
    qvalue: Optional[float] = None
    significant: bool = False


@dataclass
class ValidationReport:
    per_symbol: list = field(default_factory=list)
    pooled_trades: int = 0
    pooled_sharpe: float = 0.0
    deflated_sharpe: Optional[float] = None
    pooled_bootstrap_p: Optional[float] = None
    monte_carlo: dict = field(default_factory=dict)
    num_significant: int = 0
    verdict: str = ""


class Validator:
    def __init__(self, cost_model=None, periods: int = 252, n_folds: int = 4):
        from src.backtest.costs import CostModel
        self.cost_model = cost_model or CostModel.equities()
        self.periods = periods
        self.n_folds = n_folds

    # ------------------------------------------------------------------ #
    def _backtest(self, symbol: str, period: str, interval: str, min_score: float):
        from src.backtest.costs import CostModel
        from src.backtest.engine import Backtester
        from src.data.feed import is_crypto

        cm = CostModel.crypto() if is_crypto(symbol) else self.cost_model
        return Backtester(cost_model=cm).run_pipeline(
            symbol, period=period, interval=interval, min_score=min_score)

    def walk_forward_folds(self, result) -> list[dict]:
        """Split a backtest's equity curve + trades into ``n_folds`` time segments."""
        curve = result.equity_curve
        if curve is None or len(curve) < self.n_folds:
            return []
        bounds = np.array_split(np.arange(len(curve)), self.n_folds)
        folds = []
        trades = result.trades
        for k, idx in enumerate(bounds):
            seg = curve.iloc[idx]
            seg_ret = seg.iloc[-1] / seg.iloc[0] - 1 if len(seg) > 1 else 0.0
            start_t, end_t = seg.index[0], seg.index[-1]
            n_tr = 0
            if trades is not None and not trades.empty:
                n_tr = int(((trades["entry_time"] >= start_t) &
                            (trades["entry_time"] <= end_t)).sum())
            folds.append({
                "fold": k + 1,
                "start": str(start_t.date()), "end": str(end_t.date()),
                "return": float(seg_ret),
                "sharpe": sharpe_ratio(seg.pct_change().dropna().values, self.periods),
                "trades": n_tr,
            })
        return folds

    def validate(
        self, symbols: list[str], period: str = "2y", interval: str = "1d",
        min_score: float = 70.0,
    ) -> ValidationReport:
        report = ValidationReport()
        per_period_sharpes = []
        pooled_trade_r = []          # R-multiples across all symbols
        pooled_trade_pct = []        # per-trade % of equity across all symbols

        for sym in symbols:
            try:
                res = self._backtest(sym, period, interval, min_score)
            except Exception:
                logger.exception("validate: backtest failed for %s", sym)
                res = None
            if res is None or res.trades is None or res.trades.empty:
                report.per_symbol.append(SymbolValidation(sym, 0, 0.0, 0.0, None, None))
                continue

            daily_ret = res.equity_curve.pct_change().dropna().values
            psr = probabilistic_sharpe_ratio(daily_ret, 0.0, self.periods)
            r_mult = res.trades["r_multiple"].values
            boot_p = bootstrap_pvalue(r_mult)
            sv = SymbolValidation(
                symbol=sym, num_trades=res.num_trades,
                total_return=res.total_return, sharpe=res.sharpe,
                psr=psr, bootstrap_p=boot_p,
                folds=self.walk_forward_folds(res),
            )
            report.per_symbol.append(sv)

            if len(daily_ret) > 2 and np.std(daily_ret, ddof=1) > 0:
                per_period_sharpes.append(np.mean(daily_ret) / np.std(daily_ret, ddof=1))
            pooled_trade_r.extend(r_mult.tolist())
            # per-trade % of equity (pnl / initial capital, approximate)
            pooled_trade_pct.extend((res.trades["pnl"] / 100_000.0).tolist())

        # ---- Pooled / selection-corrected significance ----------------- #
        report.pooled_trades = len(pooled_trade_r)
        if pooled_trade_r:
            report.pooled_bootstrap_p = bootstrap_pvalue(pooled_trade_r)
            report.monte_carlo = monte_carlo_sequence(pooled_trade_pct)
        if pooled_trade_pct:
            report.pooled_sharpe = sharpe_ratio(
                np.asarray(pooled_trade_pct), periods=self.periods)
        # Deflated Sharpe: trials = number of symbols tested; trial dispersion =
        # std of per-symbol per-period Sharpes (selection bias from picking the best).
        traded = [s for s in report.per_symbol if s.num_trades > 0]
        if len(per_period_sharpes) >= 2 and pooled_trade_pct:
            trials_std = float(np.std(per_period_sharpes, ddof=1))
            pooled_pp = np.asarray(pooled_trade_pct)
            report.deflated_sharpe = deflated_sharpe_ratio(
                pooled_pp, num_trials=len(symbols),
                trials_sr_std_perperiod=trials_std, periods=self.periods)

        # ---- Multiple-testing correction across symbols ---------------- #
        pvals = [s.bootstrap_p for s in report.per_symbol]
        bh = benjamini_hochberg(pvals, alpha=0.05)
        for sv, res_bh in zip(report.per_symbol, bh):
            sv.qvalue = res_bh["qvalue"]
            sv.significant = res_bh["reject"]
        report.num_significant = sum(s.significant for s in report.per_symbol)

        report.verdict = self._verdict(report)
        return report

    @staticmethod
    def _verdict(report: ValidationReport) -> str:
        if report.pooled_trades < 30:
            return ("INSUFFICIENT EVIDENCE: too few trades (%d) for any statistic to "
                    "be meaningful — need ~30+ (ideally 100+). Treat all metrics as noise."
                    % report.pooled_trades)
        dsr = report.deflated_sharpe
        p = report.pooled_bootstrap_p
        if report.num_significant > 0 and dsr is not None and dsr > 0.95:
            return ("EDGE PLAUSIBLE: %d symbol(s) survive FDR correction and deflated "
                    "Sharpe > 0.95. Still validate on unseen data before risking capital."
                    % report.num_significant)
        if p is not None and p < 0.05:
            return ("WEAK SIGNAL: pooled bootstrap p=%.3f but does not survive "
                    "selection/deflation. Not tradeable on this evidence." % p)
        return ("NO DEMONSTRATED EDGE: results are consistent with luck after costs and "
                "multiple-testing correction. Do not deploy on this evidence.")
