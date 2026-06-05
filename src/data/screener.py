"""Universe screener — per-symbol eligibility filter.

Gate applied before the agent spends effort analyzing a symbol:

* price above ``min_price`` (default $15) — no low-priced names
* average daily volume above ``min_avg_volume`` (default 1M) — liquidity
* market cap above ``min_market_cap`` (default $3B)
* not OTC / pink-sheet listed

Price and volume are read from the OHLCV frame the agent already fetches (free);
only market cap + exchange require a yfinance lookup, which is cached per day.
Crypto symbols bypass the equity-specific filters. On any data error the screen
**fails open** (allows the symbol) so a flaky lookup never silently benches a
name that's already on the screened watchlist.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from src.data.feed import is_crypto

logger = logging.getLogger(__name__)

_OTC_MARKERS = ("OTC", "PNK", "PINK", "GREY", "EXPM")


@dataclass(frozen=True)
class ScreenCriteria:
    min_price: float = 15.0
    min_market_cap: float = 3e9
    min_avg_volume: float = 1e6


class UniverseScreener:
    def __init__(self, criteria: Optional[ScreenCriteria] = None, cache_ttl_hours: int = 24):
        self.criteria = criteria or ScreenCriteria()
        self._cache: dict[str, tuple] = {}      # symbol -> (fundamentals, fetched_at)
        self._ttl = timedelta(hours=cache_ttl_hours)

    def passes(self, symbol: str, df: Optional[pd.DataFrame] = None) -> tuple[bool, str]:
        """Return (eligible, reason). Crypto bypasses; errors fail open."""
        if is_crypto(symbol):
            return True, "crypto (equity filters n/a)"
        try:
            fund = self._fundamentals(symbol)
            # Price can come from the (accurate) close, but volume must NOT —
            # Alpaca's IEX feed reports only ~2-3% of consolidated volume, so we
            # use the full-market average from fast_info for the liquidity gate.
            price = (float(df["close"].iloc[-1]) if (df is not None and not df.empty)
                     else fund.get("price"))
            avg_vol = fund.get("avg_volume")

            if price is not None and price < self.criteria.min_price:
                return False, f"price ${price:.2f} < ${self.criteria.min_price:.0f}"
            if avg_vol is not None and avg_vol < self.criteria.min_avg_volume:
                return False, f"avg vol {avg_vol:,.0f} < {self.criteria.min_avg_volume:,.0f}"
            exch = (fund.get("exchange") or "").upper()
            if any(m in exch for m in _OTC_MARKERS):
                return False, f"OTC/pink ({exch})"
            mcap = fund.get("market_cap")
            if mcap is not None and mcap < self.criteria.min_market_cap:
                return False, f"mcap ${mcap/1e9:.1f}B < ${self.criteria.min_market_cap/1e9:.0f}B"
            return True, "ok"
        except Exception:
            logger.exception("screen failed for %s — failing open", symbol)
            return True, "screen error (fail-open)"

    # ------------------------------------------------------------------ #
    def _fundamentals(self, symbol: str) -> dict:
        cached = self._cache.get(symbol)
        now = datetime.now(timezone.utc)
        if cached and (now - cached[1]) < self._ttl:
            return cached[0]
        data = self._fetch(symbol)
        self._cache[symbol] = (data, now)
        return data

    @staticmethod
    def _fetch(symbol: str) -> dict:
        import yfinance as yf
        fi = yf.Ticker(symbol.replace("/", "-")).fast_info
        out = {}
        for key, attr in (("market_cap", "market_cap"), ("price", "last_price"),
                          ("avg_volume", "three_month_average_volume"),
                          ("exchange", "exchange")):
            try:
                out[key] = getattr(fi, attr)
            except Exception:
                out[key] = None
        return out
