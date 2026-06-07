"""Standalone analyst-rating analysis via yfinance.

Fetches Wall Street analyst recommendations and price targets for a symbol and
condenses them into a small scoring contribution plus display metadata. This
module is intentionally self-contained: it imports no other project modules so
it can be dropped into any aggregator.

Public interface:
    - :class:`AnalystResult` — dataclass holding the scored result.
    - :class:`AnalystRatings` — fetcher/scorer with a simple in-process cache.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass
class AnalystResult:
    """Result of analyzing analyst ratings for a single symbol."""

    symbol: str
    points: int
    status: str  # "ok" | "error" | "insufficient"
    rating: str  # "Strong Buy" | "Buy" | "Hold" | "Sell" | "Strong Sell" | "N/A"
    n_analysts: int
    target: float
    current: float
    upside_pct: float
    never_short: bool
    badge_color: str
    summary: str


# recommendationKey -> (display rating, base points)
_KEY_MAP = {
    "strong_buy": ("Strong Buy", 8),
    "buy": ("Buy", 5),
    "hold": ("Hold", 0),
    "sell": ("Sell", -5),
    "strong_sell": ("Strong Sell", -8),
}

# display rating -> badge color
_BADGE_MAP = {
    "Strong Buy": "green",
    "Buy": "blue",
    "Hold": "grey",
    "N/A": "grey",
    "Sell": "orange",
    "Strong Sell": "red",
}


def _rating_from_mean(mean: float) -> tuple[str, int]:
    """Derive (display rating, base points) from recommendationMean (1..5)."""
    if mean <= 1.5:
        return ("Strong Buy", 8)
    if mean <= 2.5:
        return ("Buy", 5)
    if mean <= 3.5:
        return ("Hold", 0)
    if mean <= 4.5:
        return ("Sell", -5)
    return ("Strong Sell", -8)


class AnalystRatings:
    """Fetch and score analyst ratings, with a per-instance TTL cache."""

    def __init__(self, cache_ttl: int = 86400) -> None:
        """Create the analyzer.

        Args:
            cache_ttl: Seconds to cache a per-symbol result (default 24h).
        """
        self.cache_ttl = cache_ttl
        self._cache: dict[str, tuple[float, AnalystResult]] = {}

    def analyze(self, symbol: str) -> AnalystResult:
        """Analyze analyst ratings for ``symbol`` and return an AnalystResult.

        Never raises: any failure yields a status="error" result. Crypto
        symbols (containing "/") yield status="insufficient".
        """
        # Crypto has no equity analyst coverage.
        if "/" in symbol:
            return AnalystResult(
                symbol=symbol,
                points=0,
                status="insufficient",
                rating="N/A",
                n_analysts=0,
                target=0.0,
                current=0.0,
                upside_pct=0.0,
                never_short=False,
                badge_color="grey",
                summary="n/a",
            )

        # Honor cache.
        cached = self._cache.get(symbol)
        if cached is not None:
            ts, result = cached
            if (time.time() - ts) < self.cache_ttl:
                return result

        try:
            info = yf.Ticker(symbol).info or {}

            key = info.get("recommendationKey")
            mean = info.get("recommendationMean")

            if key and key != "none" and key in _KEY_MAP:
                rating, base_points = _KEY_MAP[key]
            elif mean is not None:
                rating, base_points = _rating_from_mean(float(mean))
            else:
                rating, base_points = ("N/A", 0)

            n_analysts = int(info.get("numberOfAnalystOpinions") or 0)

            target = info.get("targetMeanPrice")
            target = float(target) if target is not None else 0.0
            current = info.get("currentPrice")
            if current is None:
                current = info.get("regularMarketPrice")
            current = float(current) if current is not None else 0.0

            # Upside and target-based bonus/penalty.
            upside_pct = 0.0
            points = base_points
            if target > 0 and current > 0:
                upside_pct = (target - current) / current * 100.0
                if upside_pct >= 15:
                    points += 5
                elif upside_pct < 0:
                    points -= 5

            never_short = rating == "Strong Buy" and n_analysts >= 10
            badge_color = _BADGE_MAP.get(rating, "grey")

            # Insufficient coverage: keep rating/target for display, zero points.
            if n_analysts < 5:
                result = AnalystResult(
                    symbol=symbol,
                    points=0,
                    status="insufficient",
                    rating=rating,
                    n_analysts=n_analysts,
                    target=target,
                    current=current,
                    upside_pct=upside_pct,
                    never_short=False,
                    badge_color=badge_color,
                    summary=f"Only {n_analysts} analysts (need 5+)",
                )
            else:
                summary = (
                    f"{rating} | {n_analysts} analysts | "
                    f"Target ${target:.0f} ({upside_pct:+.0f}%)"
                )
                result = AnalystResult(
                    symbol=symbol,
                    points=points,
                    status="ok",
                    rating=rating,
                    n_analysts=n_analysts,
                    target=target,
                    current=current,
                    upside_pct=upside_pct,
                    never_short=never_short,
                    badge_color=badge_color,
                    summary=summary,
                )

            self._cache[symbol] = (time.time(), result)
            return result

        except Exception as exc:  # noqa: BLE001 - never raise to caller
            logger.warning("analyst fetch failed for %s: %s", symbol, exc)
            return AnalystResult(
                symbol=symbol,
                points=0,
                status="error",
                rating="N/A",
                n_analysts=0,
                target=0.0,
                current=0.0,
                upside_pct=0.0,
                never_short=False,
                badge_color="grey",
                summary="analyst fetch failed",
            )


if __name__ == "__main__":
    print(AnalystRatings().analyze("AAPL"))
