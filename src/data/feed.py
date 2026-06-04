"""Market data feed for stocks and crypto, backed by Alpaca.

``MarketFeed`` exposes a single ``get_bars`` entry point that transparently
routes stock symbols (e.g. ``AAPL``) to the equities API and crypto symbols
(e.g. ``BTC/USD``) to the crypto API, returning a uniform OHLCV DataFrame.

Every fetch is wrapped so a transient API failure yields an empty DataFrame
rather than raising — the caller can simply skip the symbol this cycle.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
from alpaca.data.historical import (
    CryptoHistoricalDataClient,
    StockHistoricalDataClient,
)
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

logger = logging.getLogger(__name__)

_UNIT_MAP = {
    "min": TimeFrameUnit.Minute,
    "minute": TimeFrameUnit.Minute,
    "hour": TimeFrameUnit.Hour,
    "day": TimeFrameUnit.Day,
    "week": TimeFrameUnit.Week,
    "month": TimeFrameUnit.Month,
}

# Approximate calendar coverage per timeframe unit, used to compute a start
# date that comfortably contains `lookback` bars.
_HOURS_PER_BAR = {
    TimeFrameUnit.Minute: 1 / 60,
    TimeFrameUnit.Hour: 1,
    TimeFrameUnit.Day: 24,
    TimeFrameUnit.Week: 24 * 7,
    TimeFrameUnit.Month: 24 * 30,
}


def parse_timeframe(text: str) -> TimeFrame:
    """Parse ``"15Min"`` / ``"1Hour"`` / ``"4Hour"`` / ``"1Day"`` -> TimeFrame."""
    text = text.strip()
    num = "".join(ch for ch in text if ch.isdigit()) or "1"
    unit_str = "".join(ch for ch in text if ch.isalpha()).lower()
    if unit_str not in _UNIT_MAP:
        raise ValueError(f"Unrecognized timeframe unit in {text!r}")
    return TimeFrame(amount=int(num), unit=_UNIT_MAP[unit_str])


def is_crypto(symbol: str) -> bool:
    """Crypto pairs are written with a slash, e.g. ``BTC/USD``."""
    return "/" in symbol


class MarketFeed:
    """Fetches historical OHLCV bars for stocks and crypto."""

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        stock_feed: str = "iex",
    ) -> None:
        self._stock_client = StockHistoricalDataClient(api_key, secret_key)
        # Crypto data is public; keys are accepted but not required.
        self._crypto_client = CryptoHistoricalDataClient(api_key, secret_key)
        self._stock_feed = stock_feed

    def get_bars(
        self,
        symbol: str,
        timeframe: str | TimeFrame,
        lookback: int = 300,
    ) -> pd.DataFrame:
        """Return up to ``lookback`` recent bars as an OHLCV DataFrame.

        Columns: ``open, high, low, close, volume``; indexed by timestamp,
        sorted oldest-first. Returns an empty DataFrame on any error.
        """
        tf = parse_timeframe(timeframe) if isinstance(timeframe, str) else timeframe
        try:
            if is_crypto(symbol):
                return self._get_crypto_bars(symbol, tf, lookback)
            return self._get_stock_bars(symbol, tf, lookback)
        except Exception:
            logger.exception("Failed to fetch bars for %s", symbol)
            return pd.DataFrame()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _window_start(self, tf: TimeFrame, lookback: int) -> datetime:
        hours_per_bar = _HOURS_PER_BAR.get(tf.unit, 24) * tf.amount
        # Pad 3x to absorb weekends/holidays/market-closed gaps.
        total_hours = hours_per_bar * lookback * 3 + 24
        return datetime.now(timezone.utc) - timedelta(hours=total_hours)

    def _get_stock_bars(
        self, symbol: str, tf: TimeFrame, lookback: int
    ) -> pd.DataFrame:
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            start=self._window_start(tf, lookback),
            # Leave a buffer for the 15-min IEX delay on the free feed.
            end=datetime.now(timezone.utc) - timedelta(minutes=16),
            feed=self._stock_feed,
        )
        bars = self._stock_client.get_stock_bars(request)
        return self._frame(bars.df, symbol, lookback)

    def _get_crypto_bars(
        self, symbol: str, tf: TimeFrame, lookback: int
    ) -> pd.DataFrame:
        request = CryptoBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            start=self._window_start(tf, lookback),
        )
        bars = self._crypto_client.get_crypto_bars(request)
        return self._frame(bars.df, symbol, lookback)

    @staticmethod
    def _frame(df: pd.DataFrame | None, symbol: str, lookback: int) -> pd.DataFrame:
        if df is None or df.empty:
            logger.warning("No bars returned for %s", symbol)
            return pd.DataFrame()
        # `.df` is multi-indexed by (symbol, timestamp) even for one symbol.
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level="symbol")
        keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
        return df[keep].sort_index().tail(lookback)
