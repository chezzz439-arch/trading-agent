"""Phase 4 — Sentiment & macro context.

Fetches a market-wide risk backdrop from public ETFs/indices via yfinance: a
Fear & Greed proxy (VIX), SPY trend, sector rotation (XLK/XLF/XLE/XLV/XLI
relative strength), dollar strength (UUP), gold-vs-stocks ratio, treasuries
(TLT), and crypto relative strength (BTC vs ETH).

This is **live-only** context — it is not point-in-time reconstructable for
backtests. Every fetch degrades gracefully: a missing series yields ``None``
rather than raising, and a stale cache is reused within ``ttl`` seconds.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLI"]
_MACRO = ["^VIX", "SPY", "UUP", "GLD", "TLT", "BTC-USD", "ETH-USD"]


@dataclass
class Sentiment:
    vix: Optional[float] = None
    fear_greed: str = "unknown"        # extreme_fear / fear / neutral / greed / extreme_greed
    market_trend: str = "unknown"      # bull / bear / neutral
    sector_leaders: list = field(default_factory=list)
    sector_laggards: list = field(default_factory=list)
    dollar_strength: str = "unknown"   # strong / weak / neutral
    gold_vs_stocks: str = "unknown"    # risk_off / risk_on / neutral
    yields_trend: str = "unknown"      # rising / falling / neutral (via TLT inverse)
    crypto_leader: str = "unknown"     # btc / eth / neutral
    risk_state: str = "unknown"        # risk_on / risk_off / mixed

    def as_dict(self) -> dict:
        return self.__dict__


class SentimentAnalyzer:
    def __init__(self, ttl: int = 300):
        self.ttl = ttl
        self._cache: Optional[Sentiment] = None
        self._cache_ts: float = 0.0

    def analyze(self, now: Optional[float] = None) -> Sentiment:
        """Return current market sentiment, cached for ``ttl`` seconds.

        ``now`` (epoch seconds) may be supplied by the caller; if omitted the
        cache simply isn't time-bounded this call (we avoid time.time() in
        deterministic contexts).
        """
        if self._cache is not None and now is not None and (now - self._cache_ts) < self.ttl:
            return self._cache
        try:
            data = self._download()
            s = self._build(data)
        except Exception:
            logger.exception("SentimentAnalyzer.analyze failed")
            s = Sentiment()
        self._cache = s
        self._cache_ts = now if now is not None else time.time()
        return s

    # ------------------------------------------------------------------ #
    def _download(self) -> dict[str, pd.Series]:
        import yfinance as yf

        tickers = _MACRO + _SECTORS
        raw = yf.download(tickers, period="3mo", interval="1d",
                          auto_adjust=True, progress=False)
        closes: dict[str, pd.Series] = {}
        # yf returns a column MultiIndex (field, ticker) for multiple tickers.
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"]
            for t in tickers:
                if t in close.columns:
                    closes[t] = close[t].dropna()
        else:  # single ticker fallback
            closes[tickers[0]] = raw["Close"].dropna()
        return closes

    def _build(self, closes: dict[str, pd.Series]) -> Sentiment:
        s = Sentiment()

        # VIX / Fear & Greed proxy.
        vix_s = closes.get("^VIX")
        if vix_s is not None and len(vix_s):
            s.vix = float(vix_s.iloc[-1])
            s.fear_greed = (
                "extreme_fear" if s.vix > 32 else
                "fear" if s.vix > 24 else
                "greed" if s.vix < 14 else
                "extreme_greed" if s.vix < 11 else
                "neutral"
            )

        # SPY trend (50-EMA).
        spy = closes.get("SPY")
        if spy is not None and len(spy) >= 50:
            ema50 = spy.ewm(span=50, adjust=False).mean().iloc[-1]
            ret20 = spy.iloc[-1] / spy.iloc[-20] - 1 if len(spy) > 20 else 0
            if spy.iloc[-1] > ema50 and ret20 > 0:
                s.market_trend = "bull"
            elif spy.iloc[-1] < ema50 and ret20 < 0:
                s.market_trend = "bear"
            else:
                s.market_trend = "neutral"

        # Sector rotation by 20-day relative strength.
        perf = {}
        for sec in _SECTORS:
            ss = closes.get(sec)
            if ss is not None and len(ss) > 20:
                perf[sec] = float(ss.iloc[-1] / ss.iloc[-20] - 1)
        if perf:
            ranked = sorted(perf, key=perf.get, reverse=True)
            s.sector_leaders = ranked[:2]
            s.sector_laggards = ranked[-2:]

        # Dollar strength (UUP 20-day change).
        uup = closes.get("UUP")
        if uup is not None and len(uup) > 20:
            chg = uup.iloc[-1] / uup.iloc[-20] - 1
            s.dollar_strength = "strong" if chg > 0.01 else "weak" if chg < -0.01 else "neutral"

        # Gold vs stocks (GLD/SPY ratio trend) -> risk-off when gold outperforms.
        gld = closes.get("GLD")
        if gld is not None and spy is not None and len(gld) > 20 and len(spy) > 20:
            ratio = (gld / spy).dropna()
            if len(ratio) > 20:
                chg = ratio.iloc[-1] / ratio.iloc[-20] - 1
                s.gold_vs_stocks = "risk_off" if chg > 0.02 else "risk_on" if chg < -0.02 else "neutral"

        # Treasuries / yields proxy (TLT up => yields falling).
        tlt = closes.get("TLT")
        if tlt is not None and len(tlt) > 20:
            chg = tlt.iloc[-1] / tlt.iloc[-20] - 1
            s.yields_trend = "falling" if chg > 0.01 else "rising" if chg < -0.01 else "neutral"

        # Crypto relative strength.
        btc, eth = closes.get("BTC-USD"), closes.get("ETH-USD")
        if btc is not None and eth is not None and len(btc) > 20 and len(eth) > 20:
            btc_chg = btc.iloc[-1] / btc.iloc[-20] - 1
            eth_chg = eth.iloc[-1] / eth.iloc[-20] - 1
            s.crypto_leader = "btc" if btc_chg > eth_chg else "eth"

        # Aggregate risk state.
        risk_on_votes = sum([
            s.market_trend == "bull",
            s.fear_greed in ("greed", "extreme_greed", "neutral"),
            s.gold_vs_stocks == "risk_on",
            s.dollar_strength != "strong",
        ])
        risk_off_votes = sum([
            s.market_trend == "bear",
            s.fear_greed in ("fear", "extreme_fear"),
            s.gold_vs_stocks == "risk_off",
            s.dollar_strength == "strong",
        ])
        s.risk_state = ("risk_on" if risk_on_votes > risk_off_votes + 1 else
                        "risk_off" if risk_off_votes > risk_on_votes + 1 else "mixed")
        return s
