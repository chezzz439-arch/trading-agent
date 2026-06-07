"""Unit tests for the research aggregator (offline — sources are faked)."""

from __future__ import annotations

import time

from src.signals.analyst_ratings import AnalystResult
from src.signals.news_sentiment import NewsResult
from src.signals.sec_filings import InsiderResult
from src.signals.social_sentiment import SocialResult
from src.signals.research import EarningsInfo, ResearchEngine, ResearchReport


# --- builders -------------------------------------------------------------- #
def ins(points=0, **kw):
    d = dict(symbol="X", points=points, status="ok", buys=0, sells=0, csuite_buys=0,
             multiple_buyers=False, large_buy=False, summary="", emoji="⚪", detail="")
    d.update(kw); return InsiderResult(**d)

def an(points=0, **kw):
    d = dict(symbol="X", points=points, status="ok", rating="Hold", n_analysts=10,
             target=100.0, current=100.0, upside_pct=0.0, never_short=False,
             badge_color="grey", summary=""); d.update(kw); return AnalystResult(**d)

def nw(points=0, label="neutral", **kw):
    d = dict(symbol="X", points=points, status="ok", label=label, raw_score=0,
             top_headline="h", emoji="⚪", very_negative_recent=False,
             headlines_analyzed=5); d.update(kw); return NewsResult(**d)

def so(points=0, **kw):
    d = dict(symbol="X", points=points, status="ok", bull_pct=50.0, bear_pct=50.0,
             messages=10, high_volatility=False, summary=""); d.update(kw)
    return SocialResult(**d)


def engine(insider, analyst, news, social, earnings=EarningsInfo(days_to=40)):
    e = ResearchEngine(enabled=True)
    e.insider = type("F", (), {"analyze": lambda self, s: insider})()
    e.analyst = type("F", (), {"analyze": lambda self, s: analyst})()
    e.news = type("F", (), {"analyze": lambda self, s: news})()
    e.social = type("F", (), {"analyze": lambda self, s: social})()
    e._earnings_cache["X"] = (time.time(), earnings)   # avoid network
    return e


# --- clamping -------------------------------------------------------------- #
def test_total_clamped_positive():
    r = engine(ins(20), an(8), nw(8), so(6)).analyze("X")
    assert r.breakdown == {"insider": 20, "analyst": 8, "news": 8, "social": 6, "earnings": 0}
    assert r.total_points == 25            # raw 42 clamped to +25


def test_total_clamped_negative():
    r = engine(ins(-32), an(-8), nw(-8), so(-6)).analyze("X")
    assert r.total_points == -25           # raw -54 clamped to -25


def test_total_additive_unclamped():
    r = engine(ins(10), an(5), nw(0), so(3)).analyze("X")
    assert r.total_points == 18            # within band, passes through


# --- vetoes ---------------------------------------------------------------- #
def test_negative_news_blocks_long_only():
    r = engine(ins(), an(), nw(-4, label="negative"), so()).analyze("X")
    assert r.block_long and not r.block_short
    assert r.allows("long")[0] is False
    assert r.allows("short")[0] is True


def test_positive_news_blocks_short_only():
    r = engine(ins(), an(), nw(4, label="positive"), so()).analyze("X")
    assert r.block_short and not r.block_long
    assert r.allows("short")[0] is False
    assert r.allows("long")[0] is True


def test_strong_buy_blocks_short():
    r = engine(ins(), an(8, rating="Strong Buy", never_short=True), nw(), so()).analyze("X")
    assert r.block_short and r.allows("short")[0] is False


def test_very_negative_recent_blocks_all():
    r = engine(ins(), an(), nw(-8, label="very_negative", very_negative_recent=True), so()).analyze("X")
    assert r.block_trade
    assert r.allows("long")[0] is False and r.allows("short")[0] is False


def test_earnings_within_3_blocks_trade():
    r = engine(ins(), an(), nw(), so(), earnings=EarningsInfo(days_to=2)).analyze("X")
    assert r.block_trade and "earnings" in r.block_reason


# --- size factor ----------------------------------------------------------- #
def test_earnings_week_halves_size_and_penalizes():
    r = engine(ins(), an(), nw(), so(), earnings=EarningsInfo(days_to=5)).analyze("X")
    assert r.earnings_points == -5
    assert r.size_factor == 0.5


def test_high_social_volatility_halves_size():
    r = engine(ins(), an(), nw(), so(high_volatility=True)).analyze("X")
    assert r.size_factor == 0.5


def test_earnings_and_volatility_stack_to_quarter():
    r = engine(ins(), an(), nw(), so(high_volatility=True),
               earnings=EarningsInfo(days_to=4)).analyze("X")
    assert r.size_factor == 0.25


# --- disabled / crypto ----------------------------------------------------- #
def test_disabled_engine_is_neutral():
    e = ResearchEngine(enabled=False)
    r = e.analyze("AAPL")
    assert r.total_points == 0 and not r.block_trade and r.size_factor == 1.0


def test_crypto_short_circuits():
    r = ResearchEngine(enabled=True).analyze("BTC/USD")
    assert r.total_points == 0 and not r.block_trade
    assert r.source_status.get("news") == "n/a"


# --- earnings info --------------------------------------------------------- #
def test_earnings_flags():
    assert EarningsInfo(days_to=2).within3 and EarningsInfo(days_to=2).within7
    assert EarningsInfo(days_to=6).within7 and not EarningsInfo(days_to=6).within3
    assert not EarningsInfo(days_to=20).within7
    assert EarningsInfo(days_to=None).label == "—"
