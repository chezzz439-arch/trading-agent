"""Phase 5 — Multi-timeframe confluence.

Analyzes a symbol across several timeframes (default 15m/1h/4h/1d/1w), derives a
trend direction and momentum state on each, and scores how many timeframes
agree (0–5). Trades are only allowed at confluence >= ``min_confluence``.
Higher-timeframe bias (weekly + daily) can veto a lower-timeframe direction.

Confluence needs live multi-resolution data; it is driven by a ``MarketFeed``
and is primarily a live-trading filter (not reconstructable per-bar in the
yfinance backtest, where intraday history is limited).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

import ta
from src.data.feed import MarketFeed

logger = logging.getLogger(__name__)


@dataclass
class TimeframeView:
    timeframe: str
    direction: str          # long / short / neutral
    momentum: str           # bullish / bearish / flat
    rsi: Optional[float] = None
    key_resistance: Optional[float] = None
    key_support: Optional[float] = None


@dataclass
class MTFResult:
    symbol: str
    views: dict[str, TimeframeView] = field(default_factory=dict)
    dominant_direction: str = "neutral"
    confluence_score: int = 0       # how many timeframes agree with dominant
    htf_bias: str = "neutral"       # weekly+daily consensus
    agree: bool = False             # confluence_score >= threshold and htf not opposed


# Higher timeframes carry more weight for the bias override.
_HTF = ("1Day", "1Week")


class MTFConfluence:
    def __init__(
        self,
        feed: MarketFeed,
        timeframes: tuple[str, ...] = ("15Min", "1Hour", "4Hour", "1Day", "1Week"),
        lookback: int = 200,
        min_confluence: int = 3,
    ) -> None:
        self.feed = feed
        self.timeframes = timeframes
        self.lookback = lookback
        self.min_confluence = min_confluence

    def analyze(self, symbol: str) -> MTFResult:
        result = MTFResult(symbol=symbol)
        try:
            for tf in self.timeframes:
                view = self._view(symbol, tf)
                if view is not None:
                    result.views[tf] = view

            if not result.views:
                return result

            longs = sum(v.direction == "long" for v in result.views.values())
            shorts = sum(v.direction == "short" for v in result.views.values())
            if longs > shorts:
                result.dominant_direction = "long"
                result.confluence_score = longs
            elif shorts > longs:
                result.dominant_direction = "short"
                result.confluence_score = shorts

            # Higher-timeframe bias from daily + weekly.
            htf_dirs = [result.views[tf].direction for tf in _HTF if tf in result.views]
            if htf_dirs and all(d == "long" for d in htf_dirs):
                result.htf_bias = "long"
            elif htf_dirs and all(d == "short" for d in htf_dirs):
                result.htf_bias = "short"

            htf_opposed = (
                result.htf_bias != "neutral"
                and result.dominant_direction != "neutral"
                and result.htf_bias != result.dominant_direction
            )
            result.agree = (
                result.confluence_score >= self.min_confluence
                and result.dominant_direction != "neutral"
                and not htf_opposed
            )
            return result
        except Exception:
            logger.exception("MTFConfluence.analyze failed for %s", symbol)
            return result

    def _view(self, symbol: str, timeframe: str) -> Optional[TimeframeView]:
        try:
            df = self.feed.get_bars(symbol, timeframe, self.lookback)
            if df is None or len(df) < 55:
                return None
            c = df["close"]
            ema20 = c.ewm(span=20, adjust=False).mean().iloc[-1]
            ema50 = c.ewm(span=50, adjust=False).mean().iloc[-1]
            rsi = float(ta.momentum.RSIIndicator(c, 14).rsi().iloc[-1])
            macd_hist = ta.trend.MACD(c).macd_diff().iloc[-1]

            if c.iloc[-1] > ema20 > ema50 and rsi > 50:
                direction = "long"
            elif c.iloc[-1] < ema20 < ema50 and rsi < 50:
                direction = "short"
            else:
                direction = "neutral"

            momentum = "bullish" if macd_hist > 0 else "bearish" if macd_hist < 0 else "flat"
            return TimeframeView(
                timeframe=timeframe,
                direction=direction,
                momentum=momentum,
                rsi=rsi,
                key_resistance=float(df["high"].tail(50).max()),
                key_support=float(df["low"].tail(50).min()),
            )
        except Exception:
            logger.exception("MTF view failed for %s @ %s", symbol, timeframe)
            return None
