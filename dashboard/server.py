"""Premium standalone dashboard — read-only FastAPI server.

Serves the bot's live state as JSON + a single-page frontend (dashboard/static).
Reads the same files the agent writes (logs/agent_state.json, logs/positions.json,
logs/option_positions.json, daily agent logs) plus read-only Alpaca data and a
7-day-cached yfinance company-info layer.

STRICTLY READ-ONLY: this process never places, replaces, or cancels an order.
Run:  venv/bin/python dashboard/server.py    →  http://localhost:8765
"""
from __future__ import annotations

import json
import logging
import math
import re
import sys
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from alpaca.trading.client import TradingClient  # noqa: E402
from alpaca.trading.requests import GetOrdersRequest, GetPortfolioHistoryRequest  # noqa: E402
from alpaca.trading.enums import QueryOrderStatus  # noqa: E402

from config import settings  # noqa: E402
from src.data.feed import MarketFeed, is_crypto  # noqa: E402
from src.signals.technical import TechnicalAnalysis  # noqa: E402
from src.signals.quant import QuantAnalysis  # noqa: E402
from src.signals.regime import RegimeDetector  # noqa: E402
from src.signals.scorer import MasterScorer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s dash %(levelname)s %(message)s")
log = logging.getLogger("dashboard")

PORT = 8765
LOGS = ROOT / "logs"
STATE_FILE = LOGS / "agent_state.json"
REASONING_FILE = LOGS / "trade_reasoning.json"
COMPANY_CACHE = LOGS / "company_cache.json"

# ---- read-only clients ---------------------------------------------------- #
_trading = TradingClient(settings.ALPACA_API_KEY, settings.ALPACA_SECRET_KEY,
                         paper=settings.PAPER)
_feed = MarketFeed(settings.ALPACA_API_KEY, settings.ALPACA_SECRET_KEY, cache_ttl=300)
_tech = TechnicalAnalysis()
_quant = QuantAnalysis()
_regime = RegimeDetector()
_scorer = MasterScorer(min_score=settings.MIN_SCORE, rr_target=settings.RR_RATIO,
                       min_score_crypto=settings.MIN_SCORE_CRYPTO)

# ---- tiny TTL cache -------------------------------------------------------- #
_cache: dict[str, tuple[float, object]] = {}
_cache_lock = threading.Lock()


def cached(key: str, ttl: float, fn):
    with _cache_lock:
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < ttl:
            return hit[1]
    val = fn()
    with _cache_lock:
        _cache[key] = (time.time(), val)
    return val


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def state() -> dict:
    return _read_json(STATE_FILE, {})


# --------------------------------------------------------------------------- #
# Company info (yfinance, 7-day disk cache)
# --------------------------------------------------------------------------- #
_company_lock = threading.Lock()


def company_info(symbol: str) -> dict:
    """Name, blurb, sector, market cap, next earnings — cached 7 days."""
    if is_crypto(symbol):
        coin = symbol.split("/")[0]
        name = COIN_NAMES.get(coin, coin)
        return {"name": name,
                "blurb": COIN_BLURBS.get(coin, f"{name} — cryptocurrency, trades 24/7."),
                "sector": "Crypto", "industry": "Digital assets", "market_cap": None, "earnings_days": None}
    with _company_lock:
        cache = _read_json(COMPANY_CACHE, {})
        hit = cache.get(symbol)
        if hit and time.time() - hit.get("_ts", 0) < 7 * 86400:
            return hit
    info = {"name": symbol, "blurb": "", "sector": "", "industry": "", "market_cap": None,
            "earnings_days": None, "_ts": time.time()}
    try:
        import yfinance as yf
        t = yf.Ticker(symbol)
        raw = t.info or {}
        summary = (raw.get("longBusinessSummary") or "").strip()
        # First 1-2 sentences only — the card blurb.
        parts = re.split(r"(?<=[.!?])\s+", summary)
        info.update({
            "name": raw.get("shortName") or raw.get("longName") or symbol,
            "blurb": " ".join(parts[:2])[:340],
            "sector": raw.get("sector") or "",
            "industry": raw.get("industry") or "",
            "market_cap": raw.get("marketCap"),
        })
        try:
            cal = t.calendar
            dates = cal.get("Earnings Date") if isinstance(cal, dict) else None
            if dates:
                nxt = min(dates)
                info["earnings_days"] = max(0, (nxt - datetime.now().date()).days)
        except Exception:
            pass
    except Exception as e:  # never let yfinance break a page
        log.warning("company_info(%s): %s", symbol, e)
    with _company_lock:
        cache = _read_json(COMPANY_CACHE, {})
        cache[symbol] = info
        COMPANY_CACHE.write_text(json.dumps(cache))
    return info


# --------------------------------------------------------------------------- #
# Point-in-time analysis (deterministic parts only — used for reasoning)
# --------------------------------------------------------------------------- #
def analyze_symbol(symbol: str, side: str = "long") -> dict | None:
    """Recompute the deterministic score components + plain-English signals.

    Mirrors the bot's deterministic pipeline (technical/quant/regime/scorer).
    MTF + ML score neutral here, exactly as the project's honesty notes state.
    """
    def _run():
        df = _feed.get_bars(symbol, "1Day", settings.LOOKBACK_BARS)
        if df is None or df.empty or len(df) < 70:
            return None
        bench_sym = (settings.CRYPTO_RS_BENCHMARK
                     if is_crypto(symbol) and symbol != settings.CRYPTO_RS_BENCHMARK
                     else settings.MARKET_PROXY)
        bench = _feed.get_bars(bench_sym, "1Day", settings.LOOKBACK_BARS)
        tech = _tech.analyze(df)
        if tech is None:
            return None
        q = _quant.analyze(df, market_df=bench if bench is not None and not bench.empty else None)
        reg = _regime.detect(df, spy_df=bench if not is_crypto(symbol) else None)
        score = _scorer.score(symbol, side, technical=tech, quant=q, regime=reg)
        px = float(df["close"].iloc[-1])
        v, sig = tech.values, tech.signals
        rs20 = (q.values.get("rel_strength_20") if q else None)
        rs60 = (q.values.get("rel_strength_60") if q else None)

        lines: list[dict] = []

        def add(ok: bool | None, text: str):
            lines.append({"ok": bool(ok), "text": text})

        e21, e50 = v.get("ema21"), v.get("ema50")
        if e21 and e50:
            add(px > e21 > e50, f"Daily uptrend {'confirmed' if px > e21 > e50 else 'not confirmed'} "
                f"— price {'above' if px > e21 else 'below'} 21 EMA, 21 {'>' if e21 > e50 else '<'} 50 EMA")
        if sig.get("ema_stack_bull"):
            add(True, "EMAs fully stacked bullish (8 > 21 > 50 > 200)")
        rsi = v.get("rsi14")
        if rsi is not None:
            add(50 < rsi <= 70, f"RSI at {rsi:.0f} — "
                + ("bullish with room to run" if 50 < rsi <= 70 else
                   "overbought" if rsi > 70 else "below the bullish zone"))
        if v.get("macd_hist") is not None:
            add(sig.get("macd_bull", False),
                "MACD momentum " + ("positive" if sig.get("macd_bull") else "negative"))
        if rs20 is not None and rs60 is not None:
            bench_name = "BTC" if bench_sym == "BTC/USD" else bench_sym
            both = rs20 > 0 and rs60 > 0
            add(both, f"{'Outperforming' if both else 'Lagging'} {bench_name} "
                f"by {rs20:+.1%} (20d) / {rs60:+.1%} (60d)")
        if sig.get("volume_confirms") is not None:
            add(sig.get("volume_confirms", False),
                "Volume " + ("confirms the move" if sig.get("volume_confirms") else "below average"))
        if sig.get("adx_strong") is not None:
            add(bool(sig.get("adx_strong") and sig.get("di_bull")),
                f"Trend strength (ADX {v.get('adx') or 0:.0f}) — "
                + ("strong and bullish" if sig.get("adx_strong") and sig.get("di_bull") else "weak or mixed"))
        if sig.get("bullish_pattern"):
            add(True, "Bullish price pattern / higher highs forming")
        if reg and reg.label != "unknown":
            add(reg.trend in ("strong_trend", "weak_trend"),
                f"Market regime: {reg.label.replace('/', ' · ').replace('_', ' ')}")

        return {
            "symbol": symbol, "side": side, "price": px,
            "score": score.total, "breakdown": dict(score.breakdown),
            "notes": list(score.notes), "passed": score.passed,
            "trend_bias": tech.trend_bias, "signals": lines,
            "regime": reg.label if reg else "unknown",
            "gate": _scorer._gate(symbol),
        }
    try:
        return cached(f"analysis:{symbol}:{side}", 300, _run)
    except Exception as e:
        log.warning("analyze_symbol(%s): %s", symbol, e)
        return None


def _summary_sentence(rec: dict) -> str:
    """One plain-English line: why this trade exists."""
    bd = rec.get("breakdown") or {}
    top = sorted(((k, v) for k, v in bd.items() if isinstance(v, (int, float)) and v > 0),
                 key=lambda kv: -kv[1])[:2]
    drivers = " and ".join(k.replace("_", " ") for k, _ in top) or "its overall setup"
    r = rec.get("rr")
    rr_part = f" risking $1 to make ${r:.1f}" if r else ""
    research = rec.get("research") or {}
    rp = research.get("points")
    res_part = (f", with research adding {rp:+d} points" if isinstance(rp, (int, float)) and rp else "")
    return (f"{rec['symbol']} scored {rec.get('score', 0):.0f}/100, driven mainly by "
            f"{drivers}{res_part} —{rr_part} on a confirmed {rec.get('side', 'long')} setup.")


# --------------------------------------------------------------------------- #
# Trade reasoning store — captured the moment a position appears, kept forever
# --------------------------------------------------------------------------- #
_reasoning_lock = threading.Lock()


def _load_reasoning() -> dict:
    return _read_json(REASONING_FILE, {})


def _save_reasoning(d: dict) -> None:
    REASONING_FILE.write_text(json.dumps(d, indent=1, default=str))


def snapshot_new_trades() -> None:
    """Watcher: persist reasoning for any managed position we haven't captured,
    and mark closed ones (so reasoning survives after the position is gone)."""
    st = state()
    managed = {m["symbol"]: m for m in st.get("managed", [])}
    research_cards = st.get("research", {})
    with _reasoning_lock:
        store = _load_reasoning()
        changed = False
        for sym, m in managed.items():
            key = f"{sym}:{m.get('side', 'long')}"
            rec = store.get(key)
            if rec and rec.get("status") == "open":
                continue  # already captured
            analysis = analyze_symbol(sym, m.get("side", "long")) or {}
            entry, stop, target = m.get("entry"), m.get("current_stop"), m.get("target")
            rr = None
            try:
                if entry and stop and target and entry != stop:
                    rr = abs(target - entry) / abs(entry - stop)
            except Exception:
                pass
            rec = {
                "symbol": sym, "side": m.get("side", "long"),
                "opened_at": m.get("entry_time") or datetime.now(timezone.utc).isoformat(),
                "status": "open", "closed_at": None, "exit_pnl": None,
                "entry": entry, "stop": stop, "target": target, "rr": rr,
                "qty": m.get("remaining_qty"),
                "score": m.get("score") or analysis.get("score"),
                "breakdown": analysis.get("breakdown", {}),
                "signals": analysis.get("signals", []),
                "research": research_cards.get(sym, {}),
                "regime": m.get("regime") or analysis.get("regime"),
                "reconstructed": bool(analysis),
            }
            rec["summary"] = _summary_sentence(rec)
            store[key] = rec
            changed = True
            log.info("captured reasoning for %s (score %s)", sym, rec["score"])
        # Mark closed: open records whose symbol no longer appears in managed.
        closed_today = {c.get("symbol"): c for c in st.get("closed_today", [])}
        for key, rec in store.items():
            if rec.get("status") == "open" and rec["symbol"] not in managed:
                rec["status"] = "closed"
                rec["closed_at"] = datetime.now(timezone.utc).isoformat()
                c = closed_today.get(rec["symbol"])
                if c:
                    rec["exit_pnl"] = c.get("pnl")
                changed = True
        if changed:
            _save_reasoning(store)


def _watcher() -> None:
    while True:
        try:
            snapshot_new_trades()
        except Exception:
            log.exception("reasoning watcher")
        time.sleep(60)


# --------------------------------------------------------------------------- #
# Alpaca read-only fetchers
# --------------------------------------------------------------------------- #
def alpaca_account():
    return cached("account", 10, _trading.get_account)


def alpaca_positions():
    return cached("positions", 10, _trading.get_all_positions)


def portfolio_history(rng: str) -> dict:
    spec = {"1D": ("1D", "5Min"), "1W": ("1W", "1H"), "1M": ("1M", "1D"),
            "3M": ("3M", "1D"), "ALL": ("1A", "1D")}.get(rng, ("1M", "1D"))

    def _fetch():
        try:
            h = _trading.get_portfolio_history(
                GetPortfolioHistoryRequest(period=spec[0], timeframe=spec[1],
                                           extended_hours=True))
            pts = [{"t": ts, "equity": e} for ts, e in zip(h.timestamp, h.equity)
                   if e is not None and e > 0]
            # Daily series ends at yesterday's close — append live equity so the
            # curve always ends at the number shown in the hero.
            try:
                now_eq = float(alpaca_account().equity)
                if pts and abs(pts[-1]["equity"] - now_eq) > 0.01:
                    pts.append({"t": int(time.time()), "equity": now_eq})
            except Exception:
                pass
            return {"points": pts, "source": "alpaca"}
        except Exception as e:
            log.warning("portfolio_history(%s): %s — falling back to state", rng, e)
            hist = state().get("equity_history", [])
            return {"points": [{"t": p["t"], "equity": p["equity"]} for p in hist],
                    "source": "state"}
    return cached(f"hist:{rng}", 60, _fetch)


def closed_round_trips() -> list[dict]:
    """Reconstruct closed trades by FIFO-pairing filled orders (long-only book)."""
    def _fetch():
        try:
            orders = _trading.get_orders(GetOrdersRequest(
                status=QueryOrderStatus.CLOSED, limit=500,
                after=datetime.now(timezone.utc) - timedelta(days=120)))
        except Exception as e:
            log.warning("closed orders: %s", e)
            return []
        fills = []
        for o in orders:
            if not o.filled_at or not o.filled_avg_price or not o.filled_qty:
                continue
            fills.append({"symbol": o.symbol, "side": str(o.side).split(".")[-1].lower(),
                          "qty": float(o.filled_qty), "px": float(o.filled_avg_price),
                          "at": o.filled_at.isoformat(),
                          "option": len(o.symbol) > 10})
        fills.sort(key=lambda f: f["at"])
        inv: dict[str, list] = {}
        trips = []
        for f in fills:
            book = inv.setdefault(f["symbol"], [])
            if f["side"] == "buy":
                book.append(dict(f))
            else:
                qty = f["qty"]
                while qty > 1e-9 and book:
                    lot = book[0]
                    take = min(qty, lot["qty"])
                    mult = 100 if f["option"] else 1
                    trips.append({
                        "symbol": f["symbol"], "qty": take, "option": f["option"],
                        "entry_px": lot["px"], "exit_px": f["px"],
                        "entry_at": lot["at"], "exit_at": f["at"],
                        "pnl": round((f["px"] - lot["px"]) * take * mult, 2),
                        "pnl_pct": round((f["px"] / lot["px"] - 1) * 100, 2) if lot["px"] else 0,
                    })
                    lot["qty"] -= take
                    qty -= take
                    if lot["qty"] <= 1e-9:
                        book.pop(0)
        trips.sort(key=lambda t: t["exit_at"], reverse=True)
        return trips
    return cached("roundtrips", 120, _fetch)


# --------------------------------------------------------------------------- #
# Activity feed — recent agent log lines, in plain English
# --------------------------------------------------------------------------- #
_EVENT_RULES: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"(\S+): ENTERED score=(\d+\.?\d*) rr=([\d.]+)"), "trade",
     "Opened {0} — scored {1}/100, {2}:1 reward-to-risk"),
    (re.compile(r"(\S+): OPTION ENTERED (\S+) (\S+) x(\d+)"), "option",
     "Bought {2} {1} option on {0} ({3} contract(s))"),
    (re.compile(r"(\S+): blocked by risk: (.+)"), "skip", "Passed on {0} — risk gates: {1}"),
    (re.compile(r"(\S+): blocked by research: (.+)"), "skip", "Passed on {0} — research: {1}"),
    (re.compile(r"(\S+): crypto long held back — daily uptrend not confirmed"), "skip",
     "{0} setup forming but daily uptrend not confirmed yet"),
    (re.compile(r"(\S+): blocked by portfolio heat"), "skip",
     "Passed on {0} — total portfolio risk already at the cap"),
    (re.compile(r"Scan complete in ([\d.]+)s \((\d+) symbols, (\d+) deep-analyzed\)"), "scan",
     "Scanned {1} symbols, deep-analyzed {2} (took {0}s)"),
    (re.compile(r"Crypto: (\d+) scanned, (\d+) long-eligible"), "scan",
     "Crypto sweep: {0} pairs checked, {1} in a confirmed uptrend"),
    (re.compile(r"Market CLOSED — equities scored only"), "info",
     "Market closed — scoring only, no stock entries until the bell"),
    (re.compile(r"Kill switch|kill_switch|Trading halted", re.I), "alert", "Kill switch — trading halted"),
    (re.compile(r"(\S+): stop moved to breakeven"), "manage", "{0}: stop moved to breakeven — risk-free trade"),
    (re.compile(r"(\S+): trailing stop"), "manage", "{0}: trailing stop active, locking in gains"),
    (re.compile(r"(\S+): scaled out"), "manage", "{0}: took partial profits"),
    (re.compile(r"Agent started \| mode=(\w+)"), "info", "Agent started ({0} mode)"),
]
_LOG_LINE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ \w+ \S+: (.*)$")


def activity_feed(limit: int = 40) -> list[dict]:
    def _build():
        files = sorted(LOGS.glob("agent_2*.log"))[-2:]
        events = []
        for path in files:
            try:
                lines = path.read_text(errors="ignore").splitlines()[-1500:]
            except OSError:
                continue
            for ln in lines:
                m = _LOG_LINE.match(ln)
                if not m:
                    continue
                ts, msg = m.group(1), m.group(2)
                for rx, kind, tpl in _EVENT_RULES:
                    g = rx.search(msg)
                    if g:
                        try:
                            events.append({"t": ts, "kind": kind,
                                           "text": tpl.format(*g.groups())})
                        except (IndexError, KeyError):
                            events.append({"t": ts, "kind": kind, "text": msg})
                        break
        # Newest first; drop consecutive duplicate scan spam.
        events = events[::-1]
        out, last = [], None
        for e in events:
            if e["kind"] == "scan" and last == e["text"]:
                continue
            out.append(e)
            last = e["text"]
            if len(out) >= limit:
                break
        return out
    return cached("activity", 20, _build)


# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #
app = FastAPI(title="Trading Agent Dashboard", docs_url=None, redoc_url=None)
STATIC = Path(__file__).resolve().parent / "static"


@app.get("/api/health")
def api_health():
    st = state()
    updated = st.get("updated_at")
    stale = True
    if updated:
        try:
            age = (datetime.now(timezone.utc)
                   - datetime.fromisoformat(updated)).total_seconds()
            stale = age > settings.SCAN_INTERVAL * 2.5
        except ValueError:
            pass
    clock = cached("clock", 30, lambda: _trading.get_clock())
    return {"agent_online": not stale, "state_updated_at": updated,
            "market_open": bool(getattr(clock, "is_open", False)),
            "next_open": str(getattr(clock, "next_open", "")),
            "halted": st.get("halted", False), "paper": settings.PAPER}


@app.get("/api/overview")
def api_overview():
    st = state()
    acct = alpaca_account()
    equity = float(acct.equity)
    last_eq = float(acct.last_equity or equity)
    trips = closed_round_trips()
    wins = [t for t in trips if t["pnl"] > 0]
    return {
        "equity": equity,
        "today_pnl": round(equity - last_eq, 2),
        "today_pnl_pct": round((equity / last_eq - 1) * 100, 3) if last_eq else 0,
        "buying_power": float(acct.buying_power),
        "total_return_pct": round((equity / 100_000 - 1) * 100, 2),  # paper start
        "win_rate": round(len(wins) / len(trips) * 100, 1) if trips else None,
        "trades_closed": len(trips),
        "open_positions": len(st.get("managed", [])),
        "open_options": len(st.get("options", [])),
        "weekly_pnl": st.get("weekly_pnl"),
        "risk_state": st.get("risk_state"),
        "activity": activity_feed(),
    }


@app.get("/api/equity")
def api_equity(range: str = "1M"):
    return portfolio_history(range.upper())


def position_rows() -> list[dict]:
    """Managed positions enriched with live Alpaca prices, company info,
    and captured reasoning — shared by the Positions and Crypto pages."""
    st = state()
    managed = {m["symbol"]: m for m in st.get("managed", [])}
    live = {p.symbol: p for p in alpaca_positions()}
    with _reasoning_lock:
        store = _load_reasoning()
    out = []
    for sym, m in managed.items():
        lp = live.get(sym) or live.get(sym.replace("/", ""))
        cur = float(lp.current_price) if lp and lp.current_price else None
        qty = float(lp.qty) if lp else m.get("remaining_qty")
        entry, stop, target = m.get("entry"), m.get("current_stop"), m.get("target")
        prog = None
        if cur and entry and target and target != entry:
            prog = max(0.0, min(1.0, (cur - entry) / (target - entry)))
        info = company_info(sym)
        rec = store.get(f"{sym}:{m.get('side', 'long')}", {})
        out.append({
            "symbol": sym, "side": m.get("side"), "qty": qty,
            "entry": entry, "current": cur, "stop": stop, "target": target,
            "value": float(lp.market_value) if lp else None,
            "pnl": float(lp.unrealized_pl) if lp else None,
            "pnl_pct": float(lp.unrealized_plpc) * 100 if lp and lp.unrealized_plpc is not None else None,
            "progress": prog, "r_multiple": m.get("last_r"),
            "score": m.get("score"), "regime": m.get("regime"),
            "breakeven": m.get("breakeven"), "trailing": m.get("trailing"),
            "tranches": m.get("tranches", []), "bars_held": m.get("bars_held"),
            "company": {k: info.get(k) for k in ("name", "blurb", "sector", "market_cap")},
            "reasoning": {k: rec.get(k) for k in ("summary", "breakdown", "signals", "research", "rr")},
        })
    out.sort(key=lambda p: -(p["pnl"] or 0))
    return out


@app.get("/api/positions")
def api_positions():
    return {"positions": position_rows(), "options": state().get("options", [])}


# --------------------------------------------------------------------------- #
# Options page
# --------------------------------------------------------------------------- #
def _pretty_date(iso: str) -> str:
    try:
        d = datetime.fromisoformat(str(iso)).date()
    except ValueError:
        return str(iso)
    return d.strftime("%B ") + str(d.day)


def _dte(iso: str) -> int | None:
    try:
        return (datetime.fromisoformat(str(iso)).date() - datetime.now().date()).days
    except ValueError:
        return None


def portfolio_greeks(occ_symbols: list[str], contracts: dict[str, int]) -> dict | None:
    """Best-effort total delta/theta/vega from Alpaca option snapshots."""
    if not occ_symbols:
        return None

    def _fetch():
        from alpaca.data.historical.option import OptionHistoricalDataClient
        from alpaca.data.requests import OptionSnapshotRequest
        client = OptionHistoricalDataClient(settings.ALPACA_API_KEY,
                                            settings.ALPACA_SECRET_KEY)
        snaps = client.get_option_snapshot(
            OptionSnapshotRequest(symbol_or_symbols=occ_symbols))
        tot = {"delta": 0.0, "theta": 0.0, "vega": 0.0}
        n = 0
        for occ, snap in snaps.items():
            g = getattr(snap, "greeks", None)
            if g is None or g.delta is None:
                continue
            mult = contracts.get(occ, 1) * 100
            tot["delta"] += float(g.delta) * mult
            tot["theta"] += float(g.theta or 0) * mult
            tot["vega"] += float(g.vega or 0) * mult
            n += 1
        return {k: round(v, 2) for k, v in tot.items()} | {"contracts_priced": n} if n else None
    try:
        return cached("greeks:" + ",".join(sorted(occ_symbols)), 120, _fetch)
    except Exception as e:
        log.warning("portfolio_greeks: %s", e)
        return None


@app.get("/api/options_overview")
def api_options_overview():
    st = state()
    gate = settings.OPTIONS_MIN_SCORE

    positions = []
    for o in st.get("options", []):
        und = o.get("underlying") or o.get("symbol", "")[:6].rstrip("0123456789")
        info = company_info(und)
        dte = _dte(o.get("expiration"))
        verb = "UP" if (o.get("type") or "call").lower() == "call" else "DOWN"
        positions.append({
            **o, "dte": dte,
            "company": {k: info.get(k) for k in ("name", "blurb", "sector", "market_cap")},
            "plain_english": f"Betting {info.get('name') or und} goes {verb} "
                             f"by {_pretty_date(o.get('expiration'))}",
        })

    greeks = portfolio_greeks([o["symbol"] for o in st.get("options", []) if o.get("symbol")],
                              {o["symbol"]: int(o.get("contracts") or 1)
                               for o in st.get("options", []) if o.get("symbol")})

    # Best setups the bot is seeing right now: strongest longs vs the 80 gate.
    target_expiry = (datetime.now()
                     + timedelta(days=(settings.OPTIONS_DTE_MIN + settings.OPTIONS_DTE_MAX) // 2))
    setups = []
    for r in sorted(st.get("scores", []), key=lambda x: -(x.get("score") or 0)):
        if r.get("side") != "long" or is_crypto(r["symbol"]):
            continue
        info = company_info(r["symbol"])
        sc = r.get("score") or 0
        setups.append({
            "symbol": r["symbol"], "name": info.get("name"), "score": sc,
            "gate": gate, "ready": sc >= gate,
            "gap": round(max(0.0, gate - sc), 1),
            "plain_english": (f"Ready to fire — would buy an ATM call on {info.get('name') or r['symbol']} "
                              f"expiring around {target_expiry.strftime('%B ')}{target_expiry.day}"
                              if sc >= gate else
                              f"Betting {info.get('name') or r['symbol']} goes UP — needs "
                              f"{gate - sc:.0f} more points to trigger a call purchase"),
            "research_points": r.get("research"),
        })
        if len(setups) >= 6:
            break

    return {
        "positions": positions, "greeks": greeks, "setups": setups,
        "config": {
            "min_score": gate, "max_positions": settings.OPTIONS_MAX_POSITIONS,
            "dte_min": settings.OPTIONS_DTE_MIN, "dte_max": settings.OPTIONS_DTE_MAX,
            "risk_pct": settings.OPTIONS_RISK_PCT,
            "profit_target_pct": settings.OPTIONS_PROFIT_TARGET * 100,
            "stop_loss_pct": settings.OPTIONS_STOP_LOSS * 100,
        },
    }


# --------------------------------------------------------------------------- #
# Crypto page — computed server-side (crypto pairs aren't in state.scores)
# --------------------------------------------------------------------------- #
COIN_NAMES = {
    "BTC": "Bitcoin", "ETH": "Ethereum", "SOL": "Solana", "BNB": "Binance Coin",
    "XRP": "XRP", "ADA": "Cardano", "AVAX": "Avalanche", "DOGE": "Dogecoin",
    "MATIC": "Polygon", "DOT": "Polkadot", "LINK": "Chainlink", "UNI": "Uniswap",
    "LTC": "Litecoin", "ATOM": "Cosmos", "FIL": "Filecoin", "NEAR": "NEAR Protocol",
    "ARB": "Arbitrum", "OP": "Optimism", "APT": "Aptos", "INJ": "Injective",
    "AAVE": "Aave", "CRV": "Curve", "SUSHI": "SushiSwap",
}

COIN_BLURBS = {
    "BTC": "The original cryptocurrency and the market's benchmark asset.",
    "ETH": "Smart-contract platform powering most of DeFi and NFTs.",
    "SOL": "High-throughput layer-1 chain known for speed and low fees.",
    "BNB": "Exchange token of Binance, used for fees and its BNB Chain.",
    "XRP": "Payments-focused token built for fast cross-border settlement.",
    "ADA": "Proof-of-stake layer-1 with a research-driven roadmap.",
    "AVAX": "Layer-1 with subnets aimed at app-specific blockchains.",
    "DOGE": "The original memecoin — high beta to retail sentiment.",
    "MATIC": "Polygon — Ethereum scaling via sidechains and zk-rollups.",
    "DOT": "Polkadot — interoperability hub connecting parachains.",
    "LINK": "Chainlink — the dominant oracle network feeding data on-chain.",
    "UNI": "Uniswap — largest decentralized exchange protocol.",
    "LTC": "Litecoin — early Bitcoin fork used for cheap payments.",
    "ATOM": "Cosmos — an ecosystem of interconnected app-chains.",
    "FIL": "Filecoin — decentralized file-storage marketplace.",
    "NEAR": "Sharded layer-1 focused on developer-friendly UX.",
    "ARB": "Arbitrum — leading Ethereum layer-2 rollup.",
    "OP": "Optimism — Ethereum layer-2 powering the Superchain.",
    "APT": "Aptos — Move-language layer-1 from ex-Libra engineers.",
    "INJ": "Injective — finance-focused chain for on-chain derivatives.",
    "AAVE": "Aave — the largest decentralized lending protocol.",
    "CRV": "Curve — stablecoin-optimized DEX behind much of DeFi yield.",
    "SUSHI": "SushiSwap — community-run decentralized exchange.",
}


@app.get("/api/crypto")
def api_crypto():
    def _build():
        pairs = [s for s in settings.load_watchlist() if is_crypto(s)]
        try:
            bars = _feed.get_bars_batch(pairs, "1Day", settings.LOOKBACK_BARS)
        except Exception as e:
            log.warning("crypto batch bars: %s", e)
            bars = {}
        btc_df = bars.get(settings.CRYPTO_RS_BENCHMARK)
        spy_df = _feed.get_bars(settings.MARKET_PROXY, "1Day", settings.LOOKBACK_BARS)

        coins, uptrend_n = [], 0
        for sym in pairs:
            coin = sym.split("/")[0]
            base = {"symbol": sym, "coin": coin,
                    "name": company_info(sym).get("name") or coin,
                    "blurb": COIN_BLURBS.get(coin, f"{coin} — cryptocurrency, trades 24/7.")}
            df = bars.get(sym)
            if df is None or df.empty or len(df) < 60:
                coins.append({**base, "has_data": False, "status": "no_data"})
                continue
            closes = df["close"].astype(float)
            px = float(closes.iloc[-1])
            chg24 = float(closes.iloc[-1] / closes.iloc[-2] - 1) * 100 if len(closes) > 1 else None
            e21 = float(closes.ewm(span=21, adjust=False).mean().iloc[-1])
            e50 = float(closes.ewm(span=50, adjust=False).mean().iloc[-1])
            uptrend = px > e21 > e50

            score, trend_bias, beats_btc, rs20 = None, None, None, None
            try:
                tech = _tech.analyze(df)
                bench = (spy_df if sym == settings.CRYPTO_RS_BENCHMARK else btc_df)
                q = _quant.analyze(df, market_df=bench if bench is not None and not bench.empty else None)
                reg = _regime.detect(df)
                s = _scorer.score(sym, "long", technical=tech, quant=q, regime=reg)
                score = round(s.total, 1)
                trend_bias = tech.trend_bias
                if q:
                    rs20 = q.values.get("rel_strength_20")
                    rs60 = q.values.get("rel_strength_60")
                    if rs20 is not None and rs60 is not None:
                        beats_btc = bool(rs20 > 0 and rs60 > 0)
            except Exception as e:
                log.warning("crypto score %s: %s", sym, e)

            if uptrend:
                uptrend_n += 1
                status = "uptrend"
            elif trend_bias == "short":
                status = "short_biased"
            else:
                status = "neutral"
            coins.append({
                **base, "has_data": True, "price": px, "chg24_pct": round(chg24, 2) if chg24 is not None else None,
                "score": score, "gate": settings.MIN_SCORE_CRYPTO,
                "status": status, "uptrend": uptrend, "trend_bias": trend_bias,
                "beats_btc": beats_btc,
                "rs20_pct": round(rs20 * 100, 1) if rs20 is not None else None,
                "is_benchmark": sym == settings.CRYPTO_RS_BENCHMARK,
            })

        coins.sort(key=lambda c: (not c.get("uptrend", False), -(c.get("score") or -1)))
        with_data = [c for c in coins if c.get("has_data")]
        if uptrend_n == 0:
            headline = (f"0 of {len(with_data)} coins in a confirmed uptrend — "
                        "waiting for crypto to turn bullish before risking a dollar")
        else:
            headline = (f"{uptrend_n} of {len(with_data)} coins in a confirmed uptrend — "
                        f"longs unlock at score {settings.MIN_SCORE_CRYPTO:.0f}+")
        return {
            "coins": coins, "uptrend_count": uptrend_n,
            "with_data": len(with_data), "total": len(pairs),
            "headline": headline,
            "gate": settings.MIN_SCORE_CRYPTO,
        }
    # Cache only the expensive coin-scoring sweep; positions stay live so the
    # Crypto tab never disagrees with the Positions tab on P/L.
    payload = dict(cached("crypto_page", 180, _build))
    payload["positions"] = [p for p in position_rows() if is_crypto(p["symbol"])]
    return payload


@app.get("/api/reasoning")
def api_reasoning():
    snapshot_new_trades()   # opportunistic capture on view
    with _reasoning_lock:
        store = _load_reasoning()
    recs = sorted(store.values(), key=lambda r: r.get("opened_at") or "", reverse=True)
    return {"trades": recs}


@app.get("/api/watching")
def api_watching():
    st = state()
    research = st.get("research", {})
    rows = []
    for r in st.get("scores", []):
        sym = r["symbol"]
        info = company_info(sym)
        card = research.get(sym, {})
        reasons = []
        if card.get("analyst_rating"):
            reasons.append(f"{card['analyst_rating']}, {card.get('analyst_n', '?')} analysts")
        if card.get("insider_summary"):
            reasons.append(card["insider_summary"])
        if card.get("news_headline"):
            reasons.append(card["news_headline"][:90])
        gate = settings.MIN_SCORE_CRYPTO if is_crypto(sym) else settings.MIN_SCORE
        status = ("bullish" if r.get("passed")
                  else "neutral" if r.get("score", 0) >= gate - 10 else "avoid")
        rows.append({
            "symbol": sym, "score": r.get("score"), "passed": r.get("passed"),
            "side": r.get("side"), "research_points": r.get("research"),
            "status": status, "gate": gate,
            "name": info.get("name"), "blurb": info.get("blurb"),
            "sector": info.get("sector"), "market_cap": info.get("market_cap"),
            "earnings_days": card.get("earnings_days", info.get("earnings_days")),
            "earnings_label": card.get("earnings_label"),
            "reasons": reasons[:3],
        })
    rows.sort(key=lambda x: -(x["score"] or 0))
    return {"symbols": rows, "updated_at": st.get("updated_at")}


@app.get("/api/symbol/{symbol:path}")
def api_symbol(symbol: str):
    symbol = symbol.upper()
    analysis = analyze_symbol(symbol) or {}
    info = company_info(symbol)
    card = state().get("research", {}).get(symbol, {})

    def _chart():
        df = _feed.get_bars(symbol, "1Day", 240)
        if df is None or df.empty:
            return []
        import pandas as pd
        closes = df["close"].astype(float)
        e21 = closes.ewm(span=21, adjust=False).mean()
        e50 = closes.ewm(span=50, adjust=False).mean()
        tail = df.index[-120:]
        return [{"t": str(getattr(ix, "date", lambda: ix)()),
                 "close": round(float(closes.loc[ix]), 4),
                 "ema21": round(float(e21.loc[ix]), 4),
                 "ema50": round(float(e50.loc[ix]), 4)} for ix in tail]
    chart = cached(f"chart:{symbol}", 300, _chart)
    return {"symbol": symbol, "company": info, "analysis": analysis,
            "research": card, "chart": chart}


@app.get("/api/performance")
def api_performance():
    trips = closed_round_trips()
    hist = portfolio_history("ALL")["points"]
    equities = [p["equity"] for p in hist]
    # Max drawdown + daily Sharpe from the equity curve.
    peak, mdd = float("-inf"), 0.0
    for e in equities:
        peak = max(peak, e)
        if peak > 0:
            mdd = min(mdd, (e - peak) / peak)
    rets = [equities[i] / equities[i - 1] - 1 for i in range(1, len(equities))
            if equities[i - 1] > 0]
    sharpe = None
    if len(rets) > 5:
        mu = sum(rets) / len(rets)
        sd = math.sqrt(sum((r - mu) ** 2 for r in rets) / (len(rets) - 1))
        sharpe = round(mu / sd * math.sqrt(252), 2) if sd > 0 else None
    # Monthly returns from the daily curve.
    monthly: dict[str, list[float]] = {}
    for p in hist:
        ts = p["t"]
        dt = datetime.fromtimestamp(ts, tz=timezone.utc) if isinstance(ts, (int, float)) \
            else datetime.fromisoformat(str(ts))
        monthly.setdefault(dt.strftime("%Y-%m"), []).append(p["equity"])
    months = [{"month": k, "ret_pct": round((v[-1] / v[0] - 1) * 100, 2)}
              for k, v in sorted(monthly.items()) if v[0] > 0]
    wins = [t for t in trips if t["pnl"] > 0]
    losses = [t for t in trips if t["pnl"] <= 0]
    return {
        "win_rate": round(len(wins) / len(trips) * 100, 1) if trips else None,
        "trades": len(trips), "wins": len(wins), "losses": len(losses),
        "avg_win": round(sum(t["pnl"] for t in wins) / len(wins), 2) if wins else None,
        "avg_loss": round(sum(t["pnl"] for t in losses) / len(losses), 2) if losses else None,
        "best": max(trips, key=lambda t: t["pnl"]) if trips else None,
        "worst": min(trips, key=lambda t: t["pnl"]) if trips else None,
        "max_drawdown_pct": round(mdd * 100, 2),
        "sharpe": sharpe, "monthly": months,
        "recent": trips[:50],
        "equity": hist,
    }


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.exception_handler(Exception)
async def on_error(request, exc):
    log.exception("request failed: %s", request.url.path)
    return JSONResponse(status_code=500, content={"error": "temporarily unavailable"})


threading.Thread(target=_watcher, daemon=True).start()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
