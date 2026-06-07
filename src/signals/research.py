"""Research aggregator — fuse the four market-research sources + earnings.

Combines insider activity (SEC Form 4), analyst ratings (yfinance), news
sentiment (finviz + Yahoo) and social sentiment (StockTwits) into a single
:class:`ResearchReport` whose ``total_points`` (clamped to +/-25) is added to the
master technical score. Also derives trade **vetoes** (block long/short/trade)
and a **size factor** (earnings-week / high-volatility → smaller position).

Every source is already individually error-safe (returns ``status="error"`` and
0 points on failure); this layer additionally never raises, so the research
layer can never crash the trading loop. Disabled sources / failures contribute
0 points — the bot simply trades on its technical score.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

from src.signals.analyst_ratings import AnalystRatings, AnalystResult
from src.signals.news_sentiment import NewsResult, NewsSentiment
from src.signals.sec_filings import InsiderResult, SECInsiderActivity
from src.signals.social_sentiment import SocialResult, SocialSentiment

logger = logging.getLogger(__name__)

RESEARCH_CLAMP = 25          # max +/- research can move the total score
EARNINGS_WEEK_POINTS = -5    # penalty when earnings is within 7 days


@dataclass
class EarningsInfo:
    days_to: Optional[int] = None
    status: str = "ok"        # ok | unknown
    @property
    def within7(self) -> bool:
        return self.days_to is not None and 0 <= self.days_to <= 7
    @property
    def within3(self) -> bool:
        return self.days_to is not None and 0 <= self.days_to <= 3
    @property
    def label(self) -> str:
        if self.days_to is None:
            return "—"
        if self.days_to < 0:
            return "passed"
        return f"{self.days_to}d away"


@dataclass
class ResearchReport:
    symbol: str
    total_points: int = 0
    insider: Optional[InsiderResult] = None
    analyst: Optional[AnalystResult] = None
    news: Optional[NewsResult] = None
    social: Optional[SocialResult] = None
    earnings: EarningsInfo = field(default_factory=EarningsInfo)
    earnings_points: int = 0
    size_factor: float = 1.0
    block_trade: bool = False
    block_reason: str = ""
    block_long: bool = False
    block_short: bool = False
    source_status: dict = field(default_factory=dict)

    @property
    def breakdown(self) -> dict:
        return {
            "insider": self.insider.points if self.insider else 0,
            "analyst": self.analyst.points if self.analyst else 0,
            "news": self.news.points if self.news else 0,
            "social": self.social.points if self.social else 0,
            "earnings": self.earnings_points,
        }

    def applied_points(self, side: str) -> int:
        """Side-aware score contribution.

        ``total_points`` is a *bullishness* score (insider buying, Buy ratings,
        positive news, bullish social all push it up). For a LONG that's applied
        as-is; for a SHORT the sign is flipped, so bullish research penalises a
        short and bearish research strengthens it.
        """
        return -self.total_points if side == "short" else self.total_points

    def allows(self, side: str) -> tuple[bool, str]:
        """Whether a trade on ``side`` ("long"/"short") is permitted."""
        if self.block_trade:
            return False, self.block_reason
        if side == "long" and self.block_long:
            return False, "negative news (no longs)"
        if side == "short" and self.block_short:
            return False, "positive news / strong-buy rating (no shorts)"
        return True, ""

    def summary_lines(self) -> list[str]:
        """Human-readable lines for a Telegram trade alert."""
        out = []
        if self.news and self.news.status == "ok":
            out.append(f"📰 News: {self.news.top_headline[:60]} {self.news.emoji} "
                       f"{self.news.points:+d}pts")
        if self.analyst and self.analyst.status == "ok":
            out.append(f"👔 Analysts: {self.analyst.rating} | {self.analyst.n_analysts} "
                       f"analysts | Target ${self.analyst.target:.0f} "
                       f"({self.analyst.upside_pct:+.0f}%) {self.analyst.points:+d}pts")
        if self.insider and self.insider.status == "ok":
            out.append(f"🏦 Insiders: {self.insider.summary} {self.insider.emoji} "
                       f"{self.insider.points:+d}pts")
        if self.social and self.social.status == "ok":
            out.append(f"💬 StockTwits: {self.social.bull_pct:.0f}% Bullish "
                       f"{self.social.points:+d}pts")
        ok = "✅" if not (self.earnings.within7) else "⚠️"
        out.append(f"⚠️ Earnings: {self.earnings.label} {ok}")
        return out


class ResearchEngine:
    """Runs all four sources + earnings and fuses them into a ResearchReport."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.insider = SECInsiderActivity()
        self.analyst = AnalystRatings()
        self.news = NewsSentiment()
        self.social = SocialSentiment()
        self._earnings_cache: dict[str, tuple[float, EarningsInfo]] = {}
        self._earnings_ttl = 86400

    def analyze(self, symbol: str) -> ResearchReport:
        """Full research report for one symbol. Never raises."""
        if not self.enabled:
            return ResearchReport(symbol=symbol)
        # Crypto: external research sources don't cover Alpaca crypto pairs well —
        # short-circuit to a neutral report instead of firing doomed requests.
        if "/" in symbol:
            return ResearchReport(symbol=symbol,
                                  source_status={k: "n/a" for k in
                                                 ("insider", "analyst", "news",
                                                  "social", "earnings")})
        rep = ResearchReport(symbol=symbol)
        rep.insider = self._safe(self.insider.analyze, symbol, "insider")
        rep.analyst = self._safe(self.analyst.analyze, symbol, "analyst")
        rep.news = self._safe(self.news.analyze, symbol, "news")
        rep.social = self._safe(self.social.analyze, symbol, "social")
        rep.earnings = self._earnings(symbol)

        # ---- points ---------------------------------------------------- #
        if rep.earnings.within7:
            rep.earnings_points = EARNINGS_WEEK_POINTS
        raw = sum(rep.breakdown.values())
        rep.total_points = int(max(-RESEARCH_CLAMP, min(RESEARCH_CLAMP, raw)))

        # ---- vetoes ---------------------------------------------------- #
        if rep.news and rep.news.status == "ok":
            if rep.news.very_negative_recent:
                rep.block_trade = True
                rep.block_reason = "very negative news in last 24h"
            if rep.news.label in ("negative", "very_negative"):
                rep.block_long = True
            if rep.news.label in ("positive", "very_positive"):
                rep.block_short = True
        if rep.analyst and rep.analyst.never_short:
            rep.block_short = True
        if rep.earnings.within3:
            rep.block_trade = True
            rep.block_reason = f"earnings in {rep.earnings.days_to}d"

        # ---- size factor ---------------------------------------------- #
        if rep.earnings.within7:
            rep.size_factor *= 0.5            # halve to ~0.5% risk near earnings
        if rep.social and rep.social.high_volatility:
            rep.size_factor *= 0.5            # social volume spike → smaller size

        rep.source_status = {
            "insider": rep.insider.status if rep.insider else "error",
            "analyst": rep.analyst.status if rep.analyst else "error",
            "news": rep.news.status if rep.news else "error",
            "social": rep.social.status if rep.social else "error",
            "earnings": rep.earnings.status,
        }
        return rep

    # ------------------------------------------------------------------ #
    @staticmethod
    def _safe(fn, symbol, name):
        try:
            return fn(symbol)
        except Exception:
            logger.warning("research source %s failed for %s", name, symbol,
                           exc_info=True)
            return None

    def _earnings(self, symbol: str) -> EarningsInfo:
        now = time.time()
        hit = self._earnings_cache.get(symbol)
        if hit and now - hit[0] < self._earnings_ttl:
            return hit[1]
        info = EarningsInfo(status="unknown")
        try:
            import yfinance as yf
            df = yf.Ticker(symbol).get_earnings_dates(limit=12)
            if df is not None and not df.empty:
                today = datetime.now(timezone.utc)
                future = [d for d in df.index.to_pydatetime()
                          if (d.replace(tzinfo=d.tzinfo or timezone.utc)) >= today]
                if future:
                    nxt = min(future).date()
                    info = EarningsInfo(days_to=(nxt - date.today()).days, status="ok")
                else:
                    info = EarningsInfo(status="ok")  # known, none upcoming
        except Exception:
            logger.warning("earnings lookup failed for %s", symbol, exc_info=True)
        self._earnings_cache[symbol] = (now, info)
        return info
