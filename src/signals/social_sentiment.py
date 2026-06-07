"""StockTwits social-sentiment signal (standalone).

Fetches the public StockTwits symbol stream and turns the crowd's
Bullish/Bearish message tags from the last 24 hours into a small score
contribution and a coarse "high volatility" (unusual posting velocity) flag.

This module is intentionally self-contained: it imports nothing from the rest
of the project and depends only on the standard library + ``requests``. The
aggregator that consumes ``SocialResult`` uses ``high_volatility`` to halve
position size, so that flag is conservative by design.

Run directly to smoke-test:

    source venv/bin/activate && python src/signals/social_sentiment.py
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import requests

logger = logging.getLogger(__name__)

# StockTwits returns 403 to unidentified clients; a browser-ish UA gets 200.
_UA = {"User-Agent": "Mozilla/5.0"}
_STREAM_URL = "https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
_REQUEST_TIMEOUT = 20


@dataclass
class SocialResult:
    """Result of analyzing one symbol's StockTwits stream.

    Attributes:
        symbol: Ticker analyzed.
        points: Score contribution (-6..+6) from crowd sentiment.
        status: "ok" | "error" | "insufficient".
        bull_pct: Percent of tagged messages that are Bullish (0..100).
        bear_pct: Percent of tagged messages that are Bearish (0..100).
        messages: Total messages returned by the stream (tagged + untagged).
        high_volatility: True when posting velocity looks unusually high.
        summary: Human-readable one-liner.
    """

    symbol: str
    points: int
    status: str
    bull_pct: float
    bear_pct: float
    messages: int
    high_volatility: bool
    summary: str


class SocialSentiment:
    """Pulls StockTwits crowd sentiment and scores it.

    Args:
        cache_ttl: Seconds to reuse a cached result for a symbol (default 30m).
    """

    def __init__(self, cache_ttl: int = 1800) -> None:
        self.cache_ttl = cache_ttl
        # symbol -> (fetched_at_epoch, SocialResult)
        self._cache: dict[str, tuple[float, SocialResult]] = {}

    def analyze(self, symbol: str) -> SocialResult:
        """Analyze one symbol's recent StockTwits sentiment.

        Never raises: network/parse failures yield status="error".
        """
        # Crypto symbols use a different StockTwits naming scheme (e.g. BTC.X)
        # and the "BASE/QUOTE" form we receive won't resolve, so skip them.
        if "/" in symbol:
            return SocialResult(
                symbol=symbol,
                points=0,
                status="insufficient",
                bull_pct=0.0,
                bear_pct=0.0,
                messages=0,
                high_volatility=False,
                summary="n/a",
            )

        # Serve from cache if fresh.
        cached = self._cache.get(symbol)
        if cached is not None:
            fetched_at, result = cached
            if (time.time() - fetched_at) < self.cache_ttl:
                return result

        result = self._fetch_and_score(symbol)
        self._cache[symbol] = (time.time(), result)
        return result

    def _fetch_and_score(self, symbol: str) -> SocialResult:
        try:
            resp = requests.get(
                _STREAM_URL.format(symbol=symbol),
                headers=_UA,
                timeout=_REQUEST_TIMEOUT,
            )
        except Exception as exc:  # network error, timeout, etc.
            logger.warning("StockTwits fetch failed for %s: %s", symbol, exc)
            return self._error(symbol)

        # 403 = rate-limited / blocked; anything non-200 -> graceful error.
        if resp.status_code != 200:
            logger.warning(
                "StockTwits returned HTTP %s for %s", resp.status_code, symbol
            )
            return self._error(symbol)

        try:
            messages = resp.json().get("messages", []) or []
        except Exception as exc:  # malformed JSON
            logger.warning("StockTwits JSON parse failed for %s: %s", symbol, exc)
            return self._error(symbol)

        total_messages = len(messages)
        now = datetime.now(timezone.utc)
        cutoff_24h = now - timedelta(hours=24)
        cutoff_3h = now - timedelta(hours=3)

        bull = 0
        bear = 0
        created_times: list[datetime] = []

        for m in messages:
            created = self._parse_created(m.get("created_at"))
            if created is not None:
                created_times.append(created)
                # Skip messages older than 24h. If parsing failed (created is
                # None) we keep the message — per spec, still count it.
                if created < cutoff_24h:
                    continue

            basic = (m.get("entities", {}).get("sentiment") or {}).get("basic")
            if basic == "Bullish":
                bull += 1
            elif basic == "Bearish":
                bear += 1

        tagged = bull + bear

        # High-volatility heuristic (approximation): we cannot derive a true
        # baseline posting rate from a single stream call. We treat the stream
        # as "unusually active" only when StockTwits returns its full page of
        # 30 messages AND every one of them was posted within the last ~3 hours
        # — i.e. 30 posts in <=3h implies rapid, above-normal posting. This is
        # a coarse proxy; the aggregator uses it to halve position size.
        high_volatility = (
            total_messages >= 30
            and len(created_times) == total_messages
            and all(t >= cutoff_3h for t in created_times)
        )

        if tagged == 0:
            return SocialResult(
                symbol=symbol,
                points=0,
                status="insufficient",
                bull_pct=0.0,
                bear_pct=0.0,
                messages=total_messages,
                high_volatility=high_volatility,
                summary="No tagged sentiment",
            )

        bull_pct = bull / tagged * 100.0
        bear_pct = bear / tagged * 100.0

        if bull_pct >= 70:
            points = 6
        elif bull_pct >= 60:
            points = 3
        elif bear_pct >= 70:
            points = -6
        elif bear_pct >= 60:
            points = -3
        else:
            points = 0

        if bull_pct >= bear_pct:
            summary = f"{bull_pct:.0f}% Bullish ({total_messages} msgs)"
        else:
            summary = f"{bear_pct:.0f}% Bearish ({total_messages} msgs)"

        return SocialResult(
            symbol=symbol,
            points=points,
            status="ok",
            bull_pct=bull_pct,
            bear_pct=bear_pct,
            messages=total_messages,
            high_volatility=high_volatility,
            summary=summary,
        )

    @staticmethod
    def _parse_created(value: object) -> datetime | None:
        """Parse a StockTwits ISO-8601 timestamp into an aware datetime.

        Returns None if parsing fails (caller decides how to handle).
        """
        if not isinstance(value, str):
            return None
        try:
            # e.g. "2026-06-05T19:59:59Z"
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    @staticmethod
    def _error(symbol: str) -> SocialResult:
        return SocialResult(
            symbol=symbol,
            points=0,
            status="error",
            bull_pct=0.0,
            bear_pct=0.0,
            messages=0,
            high_volatility=False,
            summary="stocktwits fetch failed",
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(SocialSentiment().analyze("AAPL"))
