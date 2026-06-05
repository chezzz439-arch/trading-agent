"""Phase 1 — Comprehensive technical analysis.

``TechnicalAnalysis.analyze(df)`` computes a broad battery of trend, momentum,
volatility, volume, trend-strength, candlestick-pattern and support/resistance
indicators and returns a flat dict of the latest values plus a set of derived
boolean/score signals that the master scorer consumes.

Where the `ta` library provides a correct, well-tested implementation we use it;
the remaining indicators (DEMA/TEMA/HMA/ALMA, Supertrend, Klinger, Chaikin
volatility, historical volatility, pivots, Fibonacci, candle patterns) are
implemented here. Every indicator is wrapped so a bad/short series degrades to
NaN/None rather than raising.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

import ta

logger = logging.getLogger(__name__)


# ============================================================================ #
# Custom indicator helpers (not in `ta`)
# ============================================================================ #
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def dema(s: pd.Series, n: int) -> pd.Series:
    e = ema(s, n)
    return 2 * e - ema(e, n)


def tema(s: pd.Series, n: int) -> pd.Series:
    e1 = ema(s, n)
    e2 = ema(e1, n)
    e3 = ema(e2, n)
    return 3 * e1 - 3 * e2 + e3


def wma(s: pd.Series, n: int) -> pd.Series:
    weights = np.arange(1, n + 1)
    return s.rolling(n).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def hma(s: pd.Series, n: int) -> pd.Series:
    """Hull Moving Average."""
    half = max(1, n // 2)
    sqrt_n = max(1, int(np.sqrt(n)))
    return wma(2 * wma(s, half) - wma(s, n), sqrt_n)


def alma(s: pd.Series, n: int = 9, offset: float = 0.85, sigma: float = 6.0) -> pd.Series:
    """Arnaud Legoux Moving Average."""
    m = offset * (n - 1)
    sgma = n / sigma
    weights = np.array([np.exp(-((i - m) ** 2) / (2 * sgma ** 2)) for i in range(n)])
    weights /= weights.sum()
    return s.rolling(n).apply(lambda x: np.dot(x, weights), raw=True)


def historical_volatility(close: pd.Series, n: int) -> pd.Series:
    """Annualized close-to-close volatility over ``n`` bars."""
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(n).std() * np.sqrt(252)


def chaikin_volatility(high: pd.Series, low: pd.Series, n: int = 10) -> pd.Series:
    hl_ema = ema(high - low, n)
    return hl_ema.pct_change(n) * 100


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0):
    """Return (supertrend_line, direction) where direction is +1 up / -1 down."""
    atr = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], period).average_true_range()
    hl2 = (df["high"] + df["low"]) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    st = pd.Series(index=df.index, dtype=float)
    dir_ = pd.Series(index=df.index, dtype=float)
    st.iloc[0] = upper.iloc[0]
    dir_.iloc[0] = -1
    for i in range(1, len(df)):
        prev = st.iloc[i - 1]
        if df["close"].iloc[i] > prev:
            dir_.iloc[i] = 1
        elif df["close"].iloc[i] < prev:
            dir_.iloc[i] = -1
        else:
            dir_.iloc[i] = dir_.iloc[i - 1]
        if dir_.iloc[i] == 1:
            st.iloc[i] = max(lower.iloc[i], prev) if dir_.iloc[i - 1] == 1 else lower.iloc[i]
        else:
            st.iloc[i] = min(upper.iloc[i], prev) if dir_.iloc[i - 1] == -1 else upper.iloc[i]
    return st, dir_


def klinger_oscillator(df: pd.DataFrame, fast: int = 34, slow: int = 55) -> pd.Series:
    hlc = (df["high"] + df["low"] + df["close"]) / 3
    trend = np.sign(hlc.diff()).fillna(0)
    vol_force = df["volume"] * trend
    return ema(vol_force, fast) - ema(vol_force, slow)


def volume_rsi(volume: pd.Series, n: int = 14) -> pd.Series:
    delta = volume.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def vroc(volume: pd.Series, n: int = 14) -> pd.Series:
    return volume.pct_change(n) * 100


def force_index(df: pd.DataFrame, n: int = 13) -> pd.Series:
    return ema(df["close"].diff() * df["volume"], n)


# ============================================================================ #
# Support / resistance
# ============================================================================ #
def pivot_points(high: float, low: float, close: float) -> dict[str, float]:
    p = (high + low + close) / 3
    return {
        "pivot": p,
        "r1": 2 * p - low, "s1": 2 * p - high,
        "r2": p + (high - low), "s2": p - (high - low),
        "r3": high + 2 * (p - low), "s3": low - 2 * (high - p),
    }


def fibonacci_levels(swing_high: float, swing_low: float) -> dict[str, float]:
    diff = swing_high - swing_low
    retr = {f"retr_{int(r*1000)/10}": swing_high - diff * r
            for r in (0.236, 0.382, 0.5, 0.618, 0.786, 1.0)}
    ext = {f"ext_{int(e*10)/10}": swing_high + diff * (e - 1)
           for e in (1.272, 1.618, 2.618)}
    return {**retr, **ext}


def nearest_round_numbers(price: float) -> dict[str, float]:
    if price <= 0:
        return {}
    magnitude = 10 ** (int(np.floor(np.log10(price))) )
    step = magnitude / 2 if magnitude >= 10 else 1.0
    below = np.floor(price / step) * step
    return {"round_below": below, "round_above": below + step}


# ============================================================================ #
# Candlestick patterns (latest bar)
# ============================================================================ #
def candle_patterns(df: pd.DataFrame) -> dict[str, bool]:
    if len(df) < 3:
        return {}
    o, h, l, c = (df["open"], df["high"], df["low"], df["close"])
    o0, h0, l0, c0 = o.iloc[-1], h.iloc[-1], l.iloc[-1], c.iloc[-1]
    o1, c1 = o.iloc[-2], c.iloc[-2]
    o2, c2 = o.iloc[-3], c.iloc[-3]
    rng = max(h0 - l0, 1e-9)
    body = abs(c0 - o0)
    upper_wick = h0 - max(c0, o0)
    lower_wick = min(c0, o0) - l0

    bull = c0 > o0
    bear = c0 < o0
    return {
        "doji": body <= 0.1 * rng,
        "hammer": lower_wick >= 2 * body and upper_wick <= body and body > 0,
        "shooting_star": upper_wick >= 2 * body and lower_wick <= body and body > 0,
        "pin_bar": (lower_wick >= 2 * body) or (upper_wick >= 2 * body),
        "bullish_engulfing": bull and c1 < o1 and c0 >= o1 and o0 <= c1,
        "bearish_engulfing": bear and c1 > o1 and o0 >= c1 and c0 <= o1,
        "inside_bar": h0 <= h.iloc[-2] and l0 >= l.iloc[-2],
        "three_white_soldiers": (c0 > o0 and c1 > o1 and c2 > o2
                                 and c0 > c1 > c2),
        "three_black_crows": (c0 < o0 and c1 < o1 and c2 < o2
                              and c0 < c1 < c2),
        "morning_star": (c2 < o2 and abs(c1 - o1) < 0.3 * (h.iloc[-2] - l.iloc[-2] + 1e-9)
                         and c0 > o0 and c0 > (o2 + c2) / 2),
        "evening_star": (c2 > o2 and abs(c1 - o1) < 0.3 * (h.iloc[-2] - l.iloc[-2] + 1e-9)
                         and c0 < o0 and c0 < (o2 + c2) / 2),
    }


def higher_highs_lower_lows(df: pd.DataFrame, lookback: int = 20) -> dict[str, bool]:
    w = df.tail(lookback)
    if len(w) < 4:
        return {"higher_highs": False, "lower_lows": False}
    highs = w["high"].values
    lows = w["low"].values
    mid = len(w) // 2
    return {
        "higher_highs": highs[mid:].max() > highs[:mid].max(),
        "lower_lows": lows[mid:].min() < lows[:mid].min(),
    }


# ============================================================================ #
# Main analyzer
# ============================================================================ #
@dataclass
class TechnicalResult:
    values: dict[str, Any] = field(default_factory=dict)       # latest indicator values
    signals: dict[str, Any] = field(default_factory=dict)      # derived booleans/states
    levels: dict[str, float] = field(default_factory=dict)     # S/R levels
    bull_points: int = 0
    bear_points: int = 0

    @property
    def trend_bias(self) -> str:
        if self.bull_points > self.bear_points + 1:
            return "long"
        if self.bear_points > self.bull_points + 1:
            return "short"
        return "neutral"


class TechnicalAnalysis:
    """Computes the full technical battery and a derived signal summary."""

    def analyze(self, df: pd.DataFrame) -> Optional[TechnicalResult]:
        try:
            if df is None or len(df) < 60:
                return None
            df = df.copy()
            h, l, c, v = df["high"], df["low"], df["close"], df["volume"]
            price = float(c.iloc[-1])
            res = TechnicalResult()
            val, sig = res.values, res.signals

            # ---- Trend MAs ---------------------------------------------- #
            for n in (8, 13, 21, 50, 100, 200):
                val[f"ema{n}"] = _last(ema(c, n))
            for n in (20, 50, 200):
                val[f"sma{n}"] = _last(ta.trend.SMAIndicator(c, n).sma_indicator())
            val["dema20"] = _last(dema(c, 20))
            val["tema20"] = _last(tema(c, 20))
            val["hma20"] = _last(hma(c, 20))
            val["kama"] = _last(ta.momentum.KAMAIndicator(c).kama())
            val["alma"] = _last(alma(c))

            # Trend alignment scoring vs key MAs.
            above = sum(price > val[f"ema{n}"] for n in (8, 21, 50, 200)
                        if val[f"ema{n}"] is not None)
            sig["above_key_mas"] = above
            sig["ema_stack_bull"] = _ordered([val.get("ema8"), val.get("ema21"),
                                              val.get("ema50"), val.get("ema200")], desc=True)
            sig["ema_stack_bear"] = _ordered([val.get("ema8"), val.get("ema21"),
                                              val.get("ema50"), val.get("ema200")], desc=False)
            _tally(res, sig["ema_stack_bull"], sig["ema_stack_bear"])
            _tally(res, above >= 3, above <= 1)

            # ---- Momentum ----------------------------------------------- #
            for n in (7, 14, 21):
                val[f"rsi{n}"] = _last(ta.momentum.RSIIndicator(c, n).rsi())
            macd = ta.trend.MACD(c, 26, 12, 9)
            val["macd"] = _last(macd.macd())
            val["macd_signal"] = _last(macd.macd_signal())
            val["macd_hist"] = _last(macd.macd_diff())
            stoch = ta.momentum.StochasticOscillator(h, l, c, 14, 3)
            val["stoch_k"] = _last(stoch.stoch())
            val["stoch_d"] = _last(stoch.stoch_signal())
            val["williams_r"] = _last(ta.momentum.WilliamsRIndicator(h, l, c, 14).williams_r())
            val["cci"] = _last(ta.trend.CCIIndicator(h, l, c, 20).cci())
            val["roc"] = _last(ta.momentum.ROCIndicator(c, 12).roc())
            val["ultimate_osc"] = _last(ta.momentum.UltimateOscillator(h, l, c).ultimate_oscillator())
            val["awesome_osc"] = _last(ta.momentum.AwesomeOscillatorIndicator(h, l).awesome_oscillator())
            val["momentum"] = price - _at(c, -11)

            rsi14 = val.get("rsi14")
            sig["rsi_overbought"] = rsi14 is not None and rsi14 > 70
            sig["rsi_oversold"] = rsi14 is not None and rsi14 < 30
            sig["rsi_bull"] = rsi14 is not None and 50 < rsi14 <= 70
            sig["rsi_bear"] = rsi14 is not None and 30 <= rsi14 < 50
            sig["macd_bull"] = (val["macd_hist"] or 0) > 0
            sig["macd_bear"] = (val["macd_hist"] or 0) < 0
            _tally(res, sig["rsi_bull"], sig["rsi_bear"])
            _tally(res, sig["macd_bull"], sig["macd_bear"])

            # ---- Volatility --------------------------------------------- #
            for n in (7, 14, 21):
                val[f"atr{n}"] = _last(ta.volatility.AverageTrueRange(h, l, c, n).average_true_range())
            bb = ta.volatility.BollingerBands(c, 20, 2)
            val["bb_upper"] = _last(bb.bollinger_hband())
            val["bb_lower"] = _last(bb.bollinger_lband())
            val["bb_pct"] = _last(bb.bollinger_pband())
            kc = ta.volatility.KeltnerChannel(h, l, c, 20)
            val["kc_upper"] = _last(kc.keltner_channel_hband())
            val["kc_lower"] = _last(kc.keltner_channel_lband())
            dc = ta.volatility.DonchianChannel(h, l, c, 20)
            val["donchian_upper"] = _last(dc.donchian_channel_hband())
            val["donchian_lower"] = _last(dc.donchian_channel_lband())
            for n in (10, 20, 30):
                val[f"hvol{n}"] = _last(historical_volatility(c, n))
            val["chaikin_vol"] = _last(chaikin_volatility(h, l))

            # ---- Volume ------------------------------------------------- #
            val["obv"] = _last(ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume())
            val["vwap"] = _last(ta.volume.VolumeWeightedAveragePrice(h, l, c, v).volume_weighted_average_price())
            val["mfi"] = _last(ta.volume.MFIIndicator(h, l, c, v, 14).money_flow_index())
            val["cmf"] = _last(ta.volume.ChaikinMoneyFlowIndicator(h, l, c, v, 20).chaikin_money_flow())
            val["force_index"] = _last(force_index(df))
            val["volume_rsi"] = _last(volume_rsi(v))
            val["vroc"] = _last(vroc(v))
            val["klinger"] = _last(klinger_oscillator(df))
            vol_avg = v.tail(20).mean()
            sig["volume_confirms"] = float(v.iloc[-1]) > vol_avg
            sig["cmf_bull"] = (val["cmf"] or 0) > 0.05
            sig["cmf_bear"] = (val["cmf"] or 0) < -0.05
            _tally(res, sig["cmf_bull"], sig["cmf_bear"])

            # ---- Trend strength ----------------------------------------- #
            adx = ta.trend.ADXIndicator(h, l, c, 14)
            val["adx"] = _last(adx.adx())
            val["plus_di"] = _last(adx.adx_pos())
            val["minus_di"] = _last(adx.adx_neg())
            aroon = ta.trend.AroonIndicator(h, l, 25)
            val["aroon_up"] = _last(aroon.aroon_up())
            val["aroon_down"] = _last(aroon.aroon_down())
            val["psar"] = _last(ta.trend.PSARIndicator(h, l, c).psar())
            ichi = ta.trend.IchimokuIndicator(h, l)
            val["ichimoku_a"] = _last(ichi.ichimoku_a())
            val["ichimoku_b"] = _last(ichi.ichimoku_b())
            val["ichimoku_conv"] = _last(ichi.ichimoku_conversion_line())
            val["ichimoku_base"] = _last(ichi.ichimoku_base_line())
            st_line, st_dir = supertrend(df)
            val["supertrend"] = _last(st_line)
            val["supertrend_dir"] = _last(st_dir)
            sig["adx_strong"] = (val["adx"] or 0) > 25
            sig["di_bull"] = (val["plus_di"] or 0) > (val["minus_di"] or 0)
            sig["supertrend_bull"] = (val["supertrend_dir"] or 0) > 0
            _tally(res, sig["di_bull"] and sig["adx_strong"],
                   (not sig["di_bull"]) and sig["adx_strong"])
            _tally(res, sig["supertrend_bull"], not sig["supertrend_bull"])

            # ---- Patterns ----------------------------------------------- #
            patterns = candle_patterns(df)
            res.signals["patterns"] = patterns
            hhll = higher_highs_lower_lows(df)
            res.signals.update(hhll)
            bull_pat = sum(patterns.get(p, False) for p in
                           ("hammer", "bullish_engulfing", "morning_star", "three_white_soldiers"))
            bear_pat = sum(patterns.get(p, False) for p in
                           ("shooting_star", "bearish_engulfing", "evening_star", "three_black_crows"))
            sig["bullish_pattern"] = bull_pat > 0 or hhll["higher_highs"]
            sig["bearish_pattern"] = bear_pat > 0 or hhll["lower_lows"]
            _tally(res, sig["bullish_pattern"], sig["bearish_pattern"])

            # ---- Support / resistance ----------------------------------- #
            res.levels.update(pivot_points(_at(h, -2), _at(l, -2), _at(c, -2)))
            sw_high = float(h.tail(50).max())
            sw_low = float(l.tail(50).min())
            res.levels["swing_high_50"] = sw_high
            res.levels["swing_low_50"] = sw_low
            res.levels.update(fibonacci_levels(sw_high, sw_low))
            res.levels.update(nearest_round_numbers(price))

            val["price"] = price
            return res
        except Exception:
            logger.exception("TechnicalAnalysis.analyze failed")
            return None


# ---------------------------------------------------------------------------- #
# Small helpers
# ---------------------------------------------------------------------------- #
def _last(series: pd.Series) -> Optional[float]:
    try:
        v = float(series.iloc[-1])
        return v if np.isfinite(v) else None
    except (IndexError, ValueError, TypeError):
        return None


def _at(series: pd.Series, idx: int) -> Optional[float]:
    try:
        return float(series.iloc[idx])
    except (IndexError, ValueError, TypeError):
        return None


def _ordered(vals: list, desc: bool) -> bool:
    if any(x is None for x in vals):
        return False
    return all(vals[i] > vals[i + 1] for i in range(len(vals) - 1)) if desc \
        else all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))


def _tally(res: TechnicalResult, bull: bool, bear: bool) -> None:
    if bull:
        res.bull_points += 1
    if bear:
        res.bear_points += 1
