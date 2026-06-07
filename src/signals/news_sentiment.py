"""News-headline sentiment from finviz + Yahoo (yfinance).

Standalone module — imports nothing from the rest of the project. It pulls
recent headlines for a symbol from two independent sources (finviz quote page
and Yahoo Finance via ``yfinance``), runs a simple keyword-weighted sentiment
scan, and returns a :class:`NewsResult`.

Design notes:
- Each source is wrapped in its own ``try/except`` so one failing does not
  prevent the other from contributing. The method never raises.
- Results are cached per-symbol for ``cache_ttl`` seconds.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

try:
    import yfinance as yf
except Exception:  # pragma: no cover - yfinance should be installed
    yf = None  # type: ignore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword lists (case-insensitive substring match within each headline).
# ---------------------------------------------------------------------------
VERY_POSITIVE = [
    "record earnings", "beat expectations", "beats expectations",
    "raised guidance", "new contract", "fda approval", "major partnership",
    "stock buyback", "share buyback", "dividend increase", "record sales",
]
POSITIVE = [
    "growth", "upgrade", "bullish", "profit", "surge", "breakthrough",
    "expansion", "gains", "outperform", "record high",
]
VERY_NEGATIVE = [
    "missed earnings", "misses earnings", "lowered guidance", "cuts guidance",
    "sec investigation", "class action", "lawsuit", "ceo resigned",
    "ceo steps down", "product recall", "bankruptcy", "fraud", "data breach",
]
NEGATIVE = [
    "downgrade", "bearish", "loss", "decline", "miss", "concern", "warning",
    "plunge", "slump", "cuts",
]

# Each entry: (keyword, weight). Order matters only for transparency.
_WEIGHTED_KEYWORDS: List[Tuple[str, int]] = (
    [(k, 3) for k in VERY_POSITIVE]
    + [(k, 1) for k in POSITIVE]
    + [(k, -3) for k in VERY_NEGATIVE]
    + [(k, -1) for k in NEGATIVE]
)

_UA = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}

# How many of the most-recent headlines to analyze.
_MAX_HEADLINES = 40
# When a finviz timestamp cannot be parsed, treat the first N rows as "recent".
_FINVIZ_RECENT_FALLBACK = 10


@dataclass
class NewsResult:
    """Outcome of a news-sentiment scan for a single symbol."""

    symbol: str
    points: int
    status: str  # "ok" | "error" | "insufficient"
    label: str  # very_positive | positive | neutral | negative | very_negative
    raw_score: int
    top_headline: str
    emoji: str
    very_negative_recent: bool
    headlines_analyzed: int


@dataclass
class _Headline:
    """Internal representation of a single fetched headline."""

    title: str
    published: Optional[datetime]  # tz-aware UTC, or None if unknown
    recent_hint: bool = False  # source-provided "this is recent" fallback flag


class NewsSentiment:
    """Keyword-based news sentiment from finviz + Yahoo headlines."""

    def __init__(self, cache_ttl: int = 1200) -> None:
        """Create an analyzer.

        Args:
            cache_ttl: Seconds to cache a per-symbol result (default 1200 = 20m).
        """
        self.cache_ttl = cache_ttl
        self._cache: dict[str, Tuple[float, NewsResult]] = {}

    # ------------------------------------------------------------------ public
    def analyze(self, symbol: str) -> NewsResult:
        """Fetch + score recent news for ``symbol``. Never raises."""
        symbol = symbol.upper().strip()

        cached = self._cache.get(symbol)
        if cached and (time.time() - cached[0]) < self.cache_ttl:
            return cached[1]

        finviz_ok = False
        yahoo_ok = False
        headlines: List[_Headline] = []

        # Each source isolated so one failing never blocks the other.
        try:
            fv = self._fetch_finviz(symbol)
            headlines.extend(fv)
            finviz_ok = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("finviz news fetch failed for %s: %s", symbol, exc)

        try:
            yh = self._fetch_yahoo(symbol)
            headlines.extend(yh)
            yahoo_ok = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("yahoo news fetch failed for %s: %s", symbol, exc)

        # Both sources failed outright.
        if not finviz_ok and not yahoo_ok:
            result = NewsResult(
                symbol=symbol, points=0, status="error", label="neutral",
                raw_score=0, top_headline="", emoji="⚪",
                very_negative_recent=False, headlines_analyzed=0,
            )
            self._cache[symbol] = (time.time(), result)
            return result

        # Dedupe by lowercased title, keep first-seen order (finviz then yahoo).
        seen: set[str] = set()
        deduped: List[_Headline] = []
        for h in headlines:
            key = h.title.lower().strip()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(h)

        # No headlines despite a source "succeeding" -> insufficient.
        if not deduped:
            result = NewsResult(
                symbol=symbol, points=0, status="insufficient", label="neutral",
                raw_score=0, top_headline="", emoji="⚪",
                very_negative_recent=False, headlines_analyzed=0,
            )
            self._cache[symbol] = (time.time(), result)
            return result

        # Sort most-recent-first when timestamps are available; headlines with
        # no timestamp sink to the bottom but keep their relative order.
        deduped.sort(
            key=lambda h: h.published or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        analyzed = deduped[:_MAX_HEADLINES]

        raw_score = 0
        very_negative_recent = False
        scored: List[Tuple[_Headline, int]] = []  # (headline, its keyword score)
        now = datetime.now(timezone.utc)

        for h in analyzed:
            low = h.title.lower()
            h_score = 0
            for keyword, weight in _WEIGHTED_KEYWORDS:
                if keyword in low:
                    h_score += weight
                    if weight == -3:  # a VERY_NEGATIVE hit
                        if self._is_recent(h, now):
                            very_negative_recent = True
            raw_score += h_score
            scored.append((h, h_score))

        label, points = self._map_score(raw_score)
        emoji = self._emoji(label)
        top_headline = self._pick_top(scored)

        result = NewsResult(
            symbol=symbol,
            points=points,
            status="ok",
            label=label,
            raw_score=raw_score,
            top_headline=top_headline,
            emoji=emoji,
            very_negative_recent=very_negative_recent,
            headlines_analyzed=len(analyzed),
        )
        self._cache[symbol] = (time.time(), result)
        return result

    # ----------------------------------------------------------------- sources
    def _fetch_finviz(self, symbol: str) -> List[_Headline]:
        """Scrape the finviz quote page news table."""
        url = f"https://finviz.com/quote.ashx?t={symbol}"
        resp = requests.get(url, headers=_UA, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        tbl = soup.find(id="news-table")
        out: List[_Headline] = []
        if not tbl:
            return out

        # finviz rows carry a date/time cell. A row with a full date sets the
        # "current" date; subsequent time-only rows inherit it. Format examples:
        # "Jun-06-26 09:30AM" (full) or "09:30AM" (time-only, same day).
        last_date: Optional[datetime] = None
        rows = tbl.find_all("tr")
        for idx, tr in enumerate(rows):
            a = tr.find("a")
            if not a:
                continue
            title = a.text.strip()
            if not title:
                continue
            td = tr.find("td")
            published = None
            recent_hint = idx < _FINVIZ_RECENT_FALLBACK
            if td:
                published, last_date = self._parse_finviz_datetime(
                    td.get_text(strip=True), last_date
                )
            out.append(
                _Headline(title=title, published=published, recent_hint=recent_hint)
            )
        return out

    def _fetch_yahoo(self, symbol: str) -> List[_Headline]:
        """Pull headlines via yfinance (more robust than scraping Yahoo)."""
        if yf is None:
            raise RuntimeError("yfinance not available")
        news = yf.Ticker(symbol).news or []
        out: List[_Headline] = []
        for n in news:
            c = n.get("content", n) if isinstance(n, dict) else {}
            title = (c.get("title") if isinstance(c, dict) else None) or (
                n.get("title", "") if isinstance(n, dict) else ""
            )
            title = (title or "").strip()
            if not title:
                continue
            published = self._parse_yahoo_datetime(n, c)
            out.append(_Headline(title=title, published=published))
        return out

    # ----------------------------------------------------------- timestamp util
    @staticmethod
    def _parse_finviz_datetime(
        text: str, last_date: Optional[datetime]
    ) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Parse a finviz date/time cell.

        Returns ``(published, new_last_date)``. finviz uses "Mon-DD-YY HH:MMAM"
        for the first row of a day and "HH:MMAM" for subsequent same-day rows.
        Times are US/Eastern but we treat them as UTC-naive -> UTC for a
        best-effort "within 24h" recency check.
        """
        text = text.strip()
        parts = text.split()
        date_part: Optional[str] = None
        time_part: Optional[str] = None
        if len(parts) == 2:
            date_part, time_part = parts[0], parts[1]
        elif len(parts) == 1:
            # Either a lone date or a lone time.
            if "-" in parts[0]:
                date_part = parts[0]
            else:
                time_part = parts[0]

        new_last = last_date
        day: Optional[datetime] = None
        if date_part:
            try:
                day = datetime.strptime(date_part, "%b-%d-%y").replace(
                    tzinfo=timezone.utc
                )
                new_last = day
            except ValueError:
                day = last_date
        else:
            day = last_date

        if day is None:
            return None, new_last

        if time_part:
            try:
                t = datetime.strptime(time_part, "%I:%M%p").time()
                combined = datetime.combine(day.date(), t, tzinfo=timezone.utc)
                return combined, new_last
            except ValueError:
                return day, new_last
        return day, new_last

    @staticmethod
    def _parse_yahoo_datetime(n: dict, c: dict) -> Optional[datetime]:
        """Best-effort timestamp from a yfinance news item."""
        # Newer schema: content.pubDate (ISO 8601).
        if isinstance(c, dict):
            pub = c.get("pubDate") or c.get("displayTime")
            if isinstance(pub, str) and pub:
                try:
                    return datetime.fromisoformat(
                        pub.replace("Z", "+00:00")
                    ).astimezone(timezone.utc)
                except ValueError:
                    pass
        # Older schema: providerPublishTime (epoch seconds).
        epoch = n.get("providerPublishTime") if isinstance(n, dict) else None
        if isinstance(epoch, (int, float)) and epoch > 0:
            try:
                return datetime.fromtimestamp(epoch, tz=timezone.utc)
            except (ValueError, OSError):
                pass
        return None

    @staticmethod
    def _is_recent(h: _Headline, now: datetime) -> bool:
        """Whether a headline counts as published within the last 24h.

        Uses a parsed timestamp when available; otherwise falls back to the
        source's "recent" hint (finviz's first ~10 rows).
        """
        if h.published is not None:
            return (now - h.published) <= timedelta(hours=24)
        return h.recent_hint

    # ---------------------------------------------------------------- scoring
    @staticmethod
    def _map_score(raw_score: int) -> Tuple[str, int]:
        """Map an aggregate raw keyword score to (label, points)."""
        if raw_score >= 6:
            return "very_positive", 8
        if raw_score >= 2:
            return "positive", 4
        if raw_score >= -1:
            return "neutral", 0
        if raw_score >= -5:
            return "negative", -4
        return "very_negative", -8

    @staticmethod
    def _emoji(label: str) -> str:
        if label in ("positive", "very_positive"):
            return "🟢"
        if label in ("negative", "very_negative"):
            return "🔴"
        return "⚪"

    @staticmethod
    def _pick_top(scored: List[Tuple[_Headline, int]]) -> str:
        """Headline with largest absolute keyword score.

        Ties resolve to the most-recent / first (the list is already sorted
        most-recent-first). If nothing scored, return the latest headline.
        """
        if not scored:
            return ""
        best = max(scored, key=lambda pair: abs(pair[1]))
        if best[1] == 0:
            return scored[0][0].title  # nothing scored -> latest headline
        return best[0].title


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    analyzer = NewsSentiment()
    res = analyzer.analyze("AAPL")
    print(res)
    print("\nTop 5 scored headlines:")

    # Re-run the per-headline scan transparently for display only.
    fetched: List[_Headline] = []
    try:
        fetched.extend(analyzer._fetch_finviz("AAPL"))
    except Exception as exc:  # noqa: BLE001
        print(f"  (finviz fetch failed: {exc})")
    try:
        fetched.extend(analyzer._fetch_yahoo("AAPL"))
    except Exception as exc:  # noqa: BLE001
        print(f"  (yahoo fetch failed: {exc})")

    seen_titles: set[str] = set()
    unique: List[_Headline] = []
    for hd in fetched:
        k = hd.title.lower().strip()
        if k and k not in seen_titles:
            seen_titles.add(k)
            unique.append(hd)
    unique.sort(
        key=lambda h: h.published or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )

    display = []
    for hd in unique[:_MAX_HEADLINES]:
        low = hd.title.lower()
        s = sum(w for kw, w in _WEIGHTED_KEYWORDS if kw in low)
        display.append((hd.title, s))
    display.sort(key=lambda pair: abs(pair[1]), reverse=True)
    for title, s in display[:5]:
        print(f"  [{s:+d}] {title}")
