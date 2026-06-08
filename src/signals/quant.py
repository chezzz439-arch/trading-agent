"""Phase 2 — Statistical & quantitative analysis.

``QuantAnalysis.analyze(df, market_df)`` returns a per-symbol bundle of
statistical measures: Hurst exponent, autocorrelation, ADF stationarity,
rolling Sharpe/Sortino, z-scores, percentile rank, beta, linear-regression
trend quality, standard-error channels, mean-reversion half-life, Kelly
fraction, and a Monte-Carlo projection of the next N bars.

Cross-symbol tools (correlation matrix, cointegration pairs) are module-level
functions that take a dict of close-price series.

Heavy/optional dependencies (statsmodels) are imported lazily so the module
still loads if they are missing; the relevant fields degrade to ``None``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================ #
# Individual statistics
# ============================================================================ #
def hurst_exponent(series: pd.Series, max_lag: int = 50) -> Optional[float]:
    """Hurst exponent via rescaled-range / variance of lagged differences.

    >0.5 trending, <0.5 mean-reverting, ~0.5 random walk.
    """
    try:
        ts = np.asarray(series.dropna(), dtype=float)
        if len(ts) < max_lag * 2:
            max_lag = max(8, len(ts) // 2)
        lags = range(2, max_lag)
        tau = [np.std(ts[lag:] - ts[:-lag]) for lag in lags]
        tau = np.asarray(tau)
        if np.any(tau <= 0):
            return None
        slope = np.polyfit(np.log(list(lags)), np.log(tau), 1)[0]
        return float(slope)
    except Exception:
        return None


def autocorrelation(series: pd.Series, lags: int = 20) -> dict[int, float]:
    ret = series.pct_change().dropna()
    out: dict[int, float] = {}
    for lag in range(1, lags + 1):
        try:
            out[lag] = float(ret.autocorr(lag))
        except Exception:
            out[lag] = float("nan")
    return out


def adf_test(series: pd.Series) -> dict[str, Optional[float]]:
    """Augmented Dickey-Fuller test for stationarity (p<0.05 => stationary)."""
    try:
        from statsmodels.tsa.stattools import adfuller
        stat, pvalue, *_ = adfuller(series.dropna(), autolag="AIC")
        return {"adf_stat": float(stat), "adf_pvalue": float(pvalue),
                "stationary": bool(pvalue < 0.05)}
    except Exception:
        return {"adf_stat": None, "adf_pvalue": None, "stationary": None}


def rolling_sharpe(series: pd.Series, window: int, periods: int = 252) -> Optional[float]:
    ret = series.pct_change().dropna().tail(window)
    if len(ret) < 3 or ret.std() == 0:
        return None
    return float(ret.mean() / ret.std() * np.sqrt(periods))


def rolling_sortino(series: pd.Series, window: int, periods: int = 252) -> Optional[float]:
    ret = series.pct_change().dropna().tail(window)
    downside = ret[ret < 0]
    if len(ret) < 3 or downside.std() == 0 or np.isnan(downside.std()):
        return None
    return float(ret.mean() / downside.std() * np.sqrt(periods))


def zscore(series: pd.Series, window: int) -> Optional[float]:
    w = series.tail(window)
    if len(w) < 3 or w.std() == 0:
        return None
    return float((series.iloc[-1] - w.mean()) / w.std())


def percentile_rank(series: pd.Series, window: int = 252) -> Optional[float]:
    w = series.tail(window)
    if len(w) < 5:
        return None
    return float((w < series.iloc[-1]).mean() * 100)


def beta_to_market(series: pd.Series, market: pd.Series) -> Optional[float]:
    try:
        a = series.pct_change().dropna()
        b = market.pct_change().dropna()
        joined = pd.concat([a, b], axis=1, join="inner").dropna()
        if len(joined) < 20 or joined.iloc[:, 1].var() == 0:
            return None
        cov = np.cov(joined.iloc[:, 0], joined.iloc[:, 1])[0, 1]
        return float(cov / joined.iloc[:, 1].var())
    except Exception:
        return None


def relative_strength(series: pd.Series, market: Optional[pd.Series]) -> dict[str, Optional[float]]:
    """Excess return of ``series`` over the market on 20 and 60 bars.

    ``rel_strength_N = symbol_return_N - market_return_N``. ``rs_outperform`` is
    True only when the symbol beats the market on BOTH horizons — the momentum
    strategy's relative-strength filter. Returns Nones (neutral) without a market
    series so the scorer doesn't veto when SPY is unavailable.
    """
    out: dict[str, Optional[float]] = {"rel_strength_20": None, "rel_strength_60": None,
                                       "rs_outperform": None}
    if market is None:
        return out

    def _excess(n: int) -> Optional[float]:
        if len(series) <= n or len(market) <= n:
            return None
        sym_ret = float(series.iloc[-1] / series.iloc[-1 - n] - 1.0)
        mkt_ret = float(market.iloc[-1] / market.iloc[-1 - n] - 1.0)
        return sym_ret - mkt_ret

    rs20, rs60 = _excess(20), _excess(60)
    out["rel_strength_20"], out["rel_strength_60"] = rs20, rs60
    if rs20 is not None and rs60 is not None:
        out["rs_outperform"] = rs20 > 0 and rs60 > 0
    return out


def linreg_trend(series: pd.Series, window: int = 50) -> dict[str, Optional[float]]:
    """Linear-regression slope and R^2 over the last ``window`` bars."""
    try:
        from scipy import stats
        y = np.asarray(series.tail(window), dtype=float)
        x = np.arange(len(y))
        if len(y) < 5:
            return {"slope": None, "r_squared": None, "slope_pct": None}
        reg = stats.linregress(x, y)
        slope_pct = reg.slope / y.mean() * 100 if y.mean() else None
        return {"slope": float(reg.slope), "r_squared": float(reg.rvalue ** 2),
                "slope_pct": slope_pct}
    except Exception:
        return {"slope": None, "r_squared": None, "slope_pct": None}


def standard_error_channel(series: pd.Series, window: int = 50) -> dict[str, Optional[float]]:
    """Regression mid-line +/- 2 standard errors over ``window`` bars."""
    try:
        y = np.asarray(series.tail(window), dtype=float)
        x = np.arange(len(y))
        coeffs = np.polyfit(x, y, 1)
        fit = np.polyval(coeffs, x)
        se = np.sqrt(np.sum((y - fit) ** 2) / (len(y) - 2))
        mid = float(fit[-1])
        return {"channel_mid": mid, "channel_upper": mid + 2 * se,
                "channel_lower": mid - 2 * se}
    except Exception:
        return {"channel_mid": None, "channel_upper": None, "channel_lower": None}


def mean_reversion_half_life(series: pd.Series) -> Optional[float]:
    """Half-life of mean reversion via an Ornstein-Uhlenbeck regression."""
    try:
        y = series.dropna()
        lag = y.shift(1).dropna()
        delta = (y - y.shift(1)).dropna()
        joined = pd.concat([delta, lag], axis=1, join="inner").dropna()
        joined.columns = ["delta", "lag"]
        beta = np.polyfit(joined["lag"], joined["delta"], 1)[0]
        if beta >= 0:
            return None  # not mean-reverting
        return float(-np.log(2) / beta)
    except Exception:
        return None


def kalman_trend(series: pd.Series) -> Optional[float]:
    """1-D Kalman filter estimate of the underlying level (latest)."""
    try:
        obs = np.asarray(series.dropna(), dtype=float)
        if len(obs) < 5:
            return None
        x, p = obs[0], 1.0
        q, r = 1e-3, 1.0  # process / measurement noise
        for z in obs[1:]:
            p += q
            k = p / (p + r)
            x += k * (z - x)
            p *= (1 - k)
        return float(x)
    except Exception:
        return None


def kelly_fraction(win_rate: float, win_loss_ratio: float) -> float:
    """Kelly optimal fraction; clamped to [0, 1]. win_loss_ratio = avgWin/avgLoss."""
    if win_loss_ratio <= 0:
        return 0.0
    f = win_rate - (1 - win_rate) / win_loss_ratio
    return float(max(0.0, min(1.0, f)))


def monte_carlo(series: pd.Series, horizon: int = 20, runs: int = 1000,
                seed: int = 7) -> dict[str, Optional[float]]:
    """Bootstrap projection of cumulative return over the next ``horizon`` bars."""
    try:
        ret = series.pct_change().dropna().values
        if len(ret) < 20:
            return {"mc_mean": None, "mc_p05": None, "mc_p95": None, "mc_prob_up": None}
        rng = np.random.default_rng(seed)
        sampled = rng.choice(ret, size=(runs, horizon), replace=True)
        cum = (1 + sampled).prod(axis=1) - 1
        return {
            "mc_mean": float(np.mean(cum)),
            "mc_p05": float(np.percentile(cum, 5)),
            "mc_p95": float(np.percentile(cum, 95)),
            "mc_prob_up": float(np.mean(cum > 0)),
        }
    except Exception:
        return {"mc_mean": None, "mc_p05": None, "mc_p95": None, "mc_prob_up": None}


# ============================================================================ #
# Cross-symbol tools
# ============================================================================ #
def correlation_matrix(closes: dict[str, pd.Series]) -> pd.DataFrame:
    """Pairwise return-correlation matrix across symbols."""
    try:
        rets = {sym: s.pct_change() for sym, s in closes.items() if s is not None and len(s) > 5}
        if len(rets) < 2:
            return pd.DataFrame()
        return pd.DataFrame(rets).corr()
    except Exception:
        logger.exception("correlation_matrix failed")
        return pd.DataFrame()


def cointegration_pairs(closes: dict[str, pd.Series], pvalue_max: float = 0.05) -> list[dict]:
    """Engle-Granger cointegration test across all symbol pairs."""
    try:
        from statsmodels.tsa.stattools import coint
        syms = [s for s, v in closes.items() if v is not None and len(v) > 30]
        out = []
        for i in range(len(syms)):
            for j in range(i + 1, len(syms)):
                a, b = closes[syms[i]], closes[syms[j]]
                joined = pd.concat([a, b], axis=1, join="inner").dropna()
                if len(joined) < 30:
                    continue
                _, pval, _ = coint(joined.iloc[:, 0], joined.iloc[:, 1])
                if pval < pvalue_max:
                    out.append({"pair": (syms[i], syms[j]), "pvalue": float(pval)})
        return sorted(out, key=lambda d: d["pvalue"])
    except Exception:
        logger.exception("cointegration_pairs failed")
        return []


# ============================================================================ #
# Main analyzer
# ============================================================================ #
@dataclass
class QuantResult:
    values: dict = field(default_factory=dict)

    @property
    def regime_hint(self) -> str:
        h = self.values.get("hurst")
        if h is None:
            return "unknown"
        if h > 0.55:
            return "trending"
        if h < 0.45:
            return "mean_reverting"
        return "random"


class QuantAnalysis:
    def analyze(self, df: pd.DataFrame, market_df: Optional[pd.DataFrame] = None) -> Optional[QuantResult]:
        try:
            if df is None or len(df) < 30:
                return None
            c = df["close"]
            v: dict = {}
            v["hurst"] = hurst_exponent(c)
            ac = autocorrelation(c, 20)
            v["autocorr_lag1"] = ac.get(1)
            v["autocorr_mean"] = float(np.nanmean(list(ac.values()))) if ac else None
            v.update(adf_test(c))
            v["sharpe_20"] = rolling_sharpe(c, 20)
            v["sharpe_60"] = rolling_sharpe(c, 60)
            v["sortino_20"] = rolling_sortino(c, 20)
            v["sortino_60"] = rolling_sortino(c, 60)
            v["zscore_20"] = zscore(c, 20)
            v["zscore_50"] = zscore(c, 50)
            v["percentile_252"] = percentile_rank(c, 252)
            v["beta_spy"] = beta_to_market(c, market_df["close"]) if market_df is not None else None
            # Relative strength vs SPY (excess return, 20 & 60 bars) — the core
            # momentum filter. None -> neutral when no market series is supplied.
            v.update(relative_strength(c, market_df["close"] if market_df is not None else None))
            v.update(linreg_trend(c, 50))
            v.update(standard_error_channel(c, 50))
            v["half_life"] = mean_reversion_half_life(c)
            v["kalman"] = kalman_trend(c)
            v.update(monte_carlo(c, 20, 1000))
            return QuantResult(values=v)
        except Exception:
            logger.exception("QuantAnalysis.analyze failed")
            return None
