"""Live trading agent — full institutional-style pipeline.

Every 5 minutes, for each watchlist symbol, the agent runs:

    technical (P1) + quant (P2) + regime (P3) + sentiment (P4) + MTF (P5)
      + ML ensemble (P6)  ->  master 0-100 score (P7)  ->  risk gates (P8)
      ->  smart/sized entry (P9)  ->  dashboard + alerts (P10)

Only candidates scoring >= MIN_SCORE that clear every risk gate are executed.
A multi-condition kill switch (daily/weekly loss, consecutive losses) flattens
the book and halts trading. Ctrl+C shuts down gracefully. Logs go to
``logs/agent_YYYYMMDD.log``; an end-of-day report is written on shutdown.
"""

from __future__ import annotations

import logging
import os
import signal as signal_module
import sys
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from config import settings
from src.data.feed import MarketFeed, is_crypto
from src.data.screener import ScreenCriteria, UniverseScreener
from src.execution.broker import Broker
from src.monitoring.dashboard import Dashboard, DashboardState
from src.risk.portfolio_risk import PortfolioRisk
from src.risk.position_sizer import PositionSizer
from src.signals.ml_signals import MLEnsemble
from src.signals.mtf import MTFConfluence
from src.signals.quant import QuantAnalysis
from src.signals.regime import RegimeDetector
from src.signals.rr_filter import RRFilter
from src.signals.research import ResearchEngine
from src.signals.scorer import MasterScorer
from src.signals.sentiment import SentimentAnalyzer
from src.signals.strategy import Signal
from src.signals.technical import TechnicalAnalysis
from src.monitoring.state_store import StateStore
from src.monitoring.telegram_bot import TelegramNotifier
from src.execution.position_manager import (
    ManagedPosition,
    PositionManager,
    PositionStore,
)
from src.signals.options_strategy import (
    OptionPosition,
    OptionPositionStore,
    OptionsStrategy,
    crossed_up50,
    exit_decision,
)

logger = logging.getLogger("trading_agent")


def configure_logging() -> None:
    os.makedirs(settings.LOG_DIR, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    log_path = os.path.join(settings.LOG_DIR, f"agent_{date_str}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
    )


class TradingAgent:
    def __init__(self) -> None:
        self._validate_credentials()
        k, s = settings.ALPACA_API_KEY, settings.ALPACA_SECRET_KEY
        self.feed = MarketFeed(k, s, stock_feed=settings.STOCK_DATA_FEED, cache_ttl=240)
        self.technical = TechnicalAnalysis()
        self.quant = QuantAnalysis()
        self.regime_detector = RegimeDetector()
        self.sentiment = SentimentAnalyzer(ttl=settings.SCAN_INTERVAL)
        self.mtf = MTFConfluence(self.feed, timeframes=settings.MTF_TIMEFRAMES,
                                 lookback=settings.LOOKBACK_BARS,
                                 min_confluence=settings.MIN_CONFLUENCE)
        self.scorer = MasterScorer(min_score=settings.MIN_SCORE, rr_target=settings.RR_RATIO)
        # Research layer (insider/analyst/news/social + earnings). Error-safe;
        # contributes a clamped +/-25 to the score and can veto/size trades.
        self.research = ResearchEngine(enabled=settings.RESEARCH_ENABLED)
        self._source_status: dict = {}            # last seen per-source health
        self._research_view: dict[str, dict] = {} # symbol -> compact research for dashboard
        self._earnings_warned: dict[str, str] = {}  # symbol -> date warned
        self.rr_filter = RRFilter(rr_ratio=settings.RR_RATIO, atr_period=settings.ATR_PERIOD,
                                  atr_multiplier=settings.ATR_MULTIPLIER,
                                  swing_lookback=settings.SWING_LOOKBACK,
                                  path_veto=settings.RR_PATH_VETO,
                                  hybrid=settings.HYBRID_TARGET_ENABLED)
        self.portfolio = PortfolioRisk(
            max_positions=settings.MAX_CONCURRENT_POSITIONS,
            daily_loss_limit=settings.DAILY_LOSS_LIMIT,
            weekly_loss_limit=settings.WEEKLY_LOSS_LIMIT,
            max_consecutive_losses=settings.MAX_CONSECUTIVE_LOSSES,
            max_correlation=settings.MAX_CORRELATION,
            portfolio_heat_max=settings.PORTFOLIO_HEAT_MAX,
            min_score=settings.MIN_SCORE, min_rr=settings.RR_RATIO,
        )
        self.broker = Broker(k, s, paper=settings.PAPER)
        self.dashboard = Dashboard(log_dir=settings.LOG_DIR)
        # Dynamic watchlist (screened) + per-symbol metadata + eligibility gate.
        self.watchlist = settings.load_watchlist()
        self.meta = settings.load_watchlist_meta()
        self.screener = UniverseScreener(ScreenCriteria(
            min_price=settings.SCREEN_MIN_PRICE,
            min_market_cap=settings.SCREEN_MIN_MARKET_CAP,
            min_avg_volume=settings.SCREEN_MIN_AVG_VOLUME))
        # Monitoring layers (Layer 1 Streamlit reads state file; Layer 2 Telegram;
        # Layer 3 terminal dash reads state file).
        self.state_store = StateStore(log_dir=settings.LOG_DIR)
        self.notifier = TelegramNotifier()
        # Stateful P9 position manager (scale-outs, dynamic stops, time exit).
        self.position_manager = PositionManager(
            scale1_r=settings.SCALE_OUT_1_R, scale2_r=settings.SCALE_OUT_2_R,
            scale_fraction=settings.SCALE_OUT_FRACTION, breakeven_r=settings.BREAKEVEN_R,
            trail_r=settings.TRAIL_R, time_exit_bars=settings.TIME_EXIT_BARS,
            time_exit_min_r=settings.TIME_EXIT_MIN_R)
        self.position_store = PositionStore(log_dir=settings.LOG_DIR)
        self.managed: dict[str, ManagedPosition] = {}
        # Options (opt-in): route strong (>= OPTIONS_MIN_SCORE) signals to long
        # calls/puts instead of stock. Disabled -> the bot trades equities only.
        self.options: OptionsStrategy | None = None
        self.option_positions: dict[str, OptionPosition] = {}
        self.option_store = OptionPositionStore(log_dir=settings.LOG_DIR)
        self._option_live: dict[str, dict] = {}   # OCC symbol -> live premium/value
        if settings.OPTIONS_ENABLED:
            self.options = OptionsStrategy(
                k, s, paper=settings.PAPER,
                dte_min=settings.OPTIONS_DTE_MIN, dte_max=settings.OPTIONS_DTE_MAX,
                risk_pct=settings.OPTIONS_RISK_PCT,
                profit_target=settings.OPTIONS_PROFIT_TARGET,
                stop_loss=settings.OPTIONS_STOP_LOSS,
                max_positions=settings.OPTIONS_MAX_POSITIONS,
                skip_earnings=settings.OPTIONS_SKIP_EARNINGS,
                expiry_exit_days=settings.OPTIONS_EXPIRY_EXIT_DAYS)
        self._ml: dict[str, MLEnsemble] = {}     # per-symbol ensemble cache
        self._running = True
        self._open_risk: dict[str, float] = {}   # symbol -> intended dollar risk
        self._pos_pnl: dict[str, float] = {}     # last seen unrealized PnL per symbol
        self._closed_today: list[dict] = []
        self._sent: dict[str, str] = {}          # summary-type -> date/week already sent
        self._pre_scores: dict[str, float] = {}  # last pre-rank score per symbol (tiering)
        self._streamlit_proc = None
        self._scan_count = 0

    @staticmethod
    def _validate_credentials() -> None:
        if not settings.ALPACA_API_KEY or not settings.ALPACA_SECRET_KEY:
            sys.exit("Missing ALPACA_API_KEY / ALPACA_SECRET_KEY in .env")

    # ------------------------------------------------------------------ #
    def shutdown(self, *_a) -> None:
        logger.info("Shutdown requested — finishing cycle and exiting.")
        self._running = False

    def run(self) -> None:
        # --- Startup sequence -------------------------------------------- #
        mode = "PAPER" if settings.PAPER else "LIVE"
        try:
            acct = self.broker.get_account()
            if str(acct.status) not in ("AccountStatus.ACTIVE", "ACTIVE"):
                self.notifier.error_alert(f"Alpaca account not active: {acct.status}")
                sys.exit(f"Alpaca account not active: {acct.status}")
            eq = float(acct.equity)
            self.portfolio.set_day_start_equity(eq)
            self.portfolio.set_week_start_equity(eq)
            logger.info("Alpaca connected: %s equity=$%.2f", acct.status, eq)
        except SystemExit:
            raise
        except Exception as e:
            self.notifier.error_alert(f"Startup: could not connect to Alpaca: {e}")
            logger.exception("Could not read starting account")
            eq = 0.0

        # Reload any open positions' lifecycle state from before a restart.
        self.managed = self.position_store.load()
        if self.managed:
            logger.info("Reloaded %d managed position(s) from disk", len(self.managed))
        # Restore the portfolio-heat counter from reloaded positions, otherwise a
        # restart resets open risk to zero and the heat cap stops protecting until
        # positions re-enter (which let the weekend order-pileup happen).
        for sym, mp in self.managed.items():
            self._open_risk[sym] = mp.risk_per_share * mp.initial_qty
        if self.options is not None:
            self.option_positions = self.option_store.load()
            for op in self.option_positions.values():
                self._open_risk[op.symbol] = op.cost_basis
            logger.info("Options ENABLED | gate>=%.0f | reloaded %d option position(s)",
                        settings.OPTIONS_MIN_SCORE, len(self.option_positions))

        self._start_streamlit()
        self.notifier.startup(mode, eq)
        logger.info("Agent started | mode=%s | %d symbols | interval=%ds | min_score=%.0f",
                    mode, len(self.watchlist), settings.SCAN_INTERVAL, settings.MIN_SCORE)
        consecutive_errors = 0
        while self._running:
            try:
                self.scan_once()
                consecutive_errors = 0
            except Exception as e:
                logger.exception("Scan cycle failed")
                consecutive_errors += 1
                # Alert once on first failure (likely Alpaca/connection loss).
                if consecutive_errors == 1:
                    self.notifier.error_alert(f"Scan cycle failed: {type(e).__name__}: {e}")
            self._scan_count += 1
            if not self._running:
                break
            self._sleep(settings.SCAN_INTERVAL)
        self._on_exit()

    def _start_streamlit(self) -> None:
        """Launch the Streamlit dashboard as a background subprocess."""
        if not getattr(settings, "STREAMLIT_AUTOSTART", True):
            return
        try:
            import subprocess
            self._streamlit_proc = subprocess.Popen(
                [sys.executable, "-m", "streamlit", "run",
                 "src/monitoring/dashboard_app.py",
                 "--server.port", "8501", "--server.headless", "true"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            logger.info("Streamlit dashboard launching at http://localhost:8501")
        except Exception:
            logger.exception("Could not launch Streamlit (run it manually)")

    def _sleep(self, seconds: int) -> None:
        for _ in range(seconds):
            if not self._running:
                return
            time.sleep(1)

    def _on_exit(self) -> None:
        try:
            self.dashboard.daily_report(self._build_state([], []))
        except Exception:
            logger.exception("daily report on exit failed")
        if self._streamlit_proc is not None:
            try:
                self._streamlit_proc.terminate()
            except Exception:
                pass
        logger.info("Agent stopped.")

    # ------------------------------------------------------------------ #
    def scan_once(self) -> None:
        _t0 = time.time()
        equity = self.broker.get_equity()
        buying_power = self.broker.get_buying_power()
        day_start = self.portfolio._day_start_equity or equity

        # Dashboard HALT button (cross-process control flag).
        halt_reason = self.state_store.halt_requested()
        if halt_reason and not self.portfolio.halted:
            self.portfolio.halted = True
            self.broker.close_all()
            self.notifier.kill_switch(reason=halt_reason, daily_loss=day_start - equity)
            self.dashboard.alert("kill_switch", halt_reason)
            self.state_store.clear_halt()

        if self.portfolio.kill_switch_triggered(equity):
            self.broker.close_all()
            self.notifier.kill_switch(reason="risk limit breached",
                                      daily_loss=day_start - equity)
            self.dashboard.alert("kill_switch", "Trading halted — book flattened")
            self._write_state([], [], equity, buying_power, None)
            return

        sentiment = self.sentiment.analyze(now=time.time())
        spy_df = self.feed.get_bars(settings.MARKET_PROXY, "1Day", settings.LOOKBACK_BARS)

        positions = self.broker.get_positions()
        open_symbols = {p.symbol for p in positions}
        # P9: manage existing positions (scale-outs, dynamic stops, exits) first.
        self._manage_positions(positions, equity)
        self._detect_closed_trades(positions, equity)
        # Options lifecycle (opt-in): re-price open options, take profit / stop /
        # expiry exits, and fire milestone alerts.
        self._manage_options(equity)
        # Ensure every held position contributes to the portfolio-heat counter,
        # including ones not tracked by the position manager (pre-existing).
        self._sync_held_risk(positions, equity)
        self._check_earnings_warnings(positions)
        total_risk = sum(self._open_risk.values())
        logger.info("Portfolio heat: $%.0f / $%.0f (%.1f%%) across %d position(s)",
                    total_risk, equity * settings.PORTFOLIO_HEAT_MAX,
                    (total_risk / equity * 100) if equity else 0.0,
                    len(self._open_risk))
        held_closes = self._held_closes(open_symbols)

        order_syms = self.broker.open_order_symbols()
        # Pending entry orders must count toward the position cap too — otherwise
        # (especially when nothing fills, e.g. weekends) the agent keeps queuing
        # past MAX_CONCURRENT_POSITIONS because only *filled* positions are
        # counted. ``committed`` = held positions + symbols with a working order.
        committed = set(open_symbols) | set(order_syms)

        # --- Tiered selection: scan hot names every cycle, cool ones less --- #
        # (pre-rank score is 0-35; ~21 ≈ "60+", ~14 ≈ "40+"). Open positions and
        # never-scored symbols are always scanned.
        cyc = self._scan_count
        scan_set = []
        for sym in self.watchlist:
            held = sym in open_symbols or sym.replace("/", "") in open_symbols
            sc = self._pre_scores.get(sym)
            # Crypto trades 24/7 and is the only thing live when equities are
            # closed, so it's always tier-1 (never demoted to the every-Nth-cycle
            # tiers even when short-biased and scoring 0).
            if held or sc is None or sc >= 21 or is_crypto(sym):
                scan_set.append(sym)                      # Tier 1 / always
            elif sc >= 14 and cyc % 2 == 0:
                scan_set.append(sym)                      # Tier 2: every 2nd
            elif cyc % 3 == 0:
                scan_set.append(sym)                      # Tier 3: every 3rd

        # --- Batch-fetch all daily bars for this cycle in one request ----- #
        bars = self.feed.get_bars_batch(scan_set, "1Day", settings.LOOKBACK_BARS)

        # --- Pre-rank: cheap technical-only pass (reusing the batched bars) - #
        candidates = []
        for symbol in scan_set:
            try:
                c = self._prerank(symbol, open_symbols, order_syms, df=bars.get(symbol))
                self._pre_scores[symbol] = c["pre_score"] if c else 0.0
                if c is not None:
                    candidates.append(c)
            except Exception:
                logger.exception("Pre-rank failed for %s", symbol)
        candidates.sort(key=lambda c: c["pre_score"], reverse=True)
        top = candidates[: settings.PRERANK_TOP_N]
        logger.info("Pre-ranked %d candidates from %d/%d scanned this cycle (tiered); "
                    "deep-analyzing top %d", len(candidates), len(scan_set),
                    len(self.watchlist), len(top))
        # Crypto visibility: it's scanned 24/7 but only enters on a long bias
        # (Alpaca can't short crypto). Surface how many were scanned vs long-
        # eligible so a quiet crypto book is explainable, not invisible.
        crypto_scanned = [s for s in scan_set if is_crypto(s)]
        crypto_long = [c["symbol"] for c in candidates if is_crypto(c["symbol"])]
        if crypto_scanned:
            logger.info("Crypto: %d scanned, %d long-eligible%s", len(crypto_scanned),
                        len(crypto_long), (" (" + ", ".join(crypto_long) + ")") if crypto_long
                        else " (all short-biased — no longs)")

        # --- Full pass: expensive pipeline (quant/MTF/ML) only on top N --- #
        scores_for_dash = []
        for c in top:
            try:
                row = self._full_evaluate(c, equity, committed, held_closes,
                                          sentiment, spy_df)
                if row is not None:
                    scores_for_dash.append(row)
            except Exception:
                logger.exception("Error evaluating %s", c["symbol"])

        state = self._build_state(positions, scores_for_dash, equity, buying_power, sentiment)
        self.dashboard.print(state)
        self._write_state(positions, scores_for_dash, equity, buying_power, sentiment)
        self._scheduled_summaries(equity, day_start, sentiment)
        logger.info("Scan complete in %.1fs (%d symbols, %d deep-analyzed)",
                    time.time() - _t0, len(self.watchlist), len(scores_for_dash))

    def _prerank(self, symbol, open_symbols, order_syms, df=None):
        """Cheap pass: bars + screen + technical -> a fast ranking score.

        Reuses the batch-fetched ``df`` when provided. Returns a candidate dict
        (carrying df + technical so the full pass doesn't recompute) or None.
        """
        if symbol in open_symbols or symbol.replace("/", "") in open_symbols:
            return None
        if symbol in order_syms or symbol.replace("/", "") in order_syms:
            return None

        if df is None:
            df = self.feed.get_bars(symbol, "1Day", settings.LOOKBACK_BARS)
        if df is None or df.empty or len(df) < 60:
            return None

        eligible, reason = self.screener.passes(symbol, df)
        if not eligible:
            logger.info("%s: screened out — %s", symbol, reason)
            return None

        tech = self.technical.analyze(df)
        if tech is None or tech.trend_bias == "neutral":
            return None
        side = tech.trend_bias
        if settings.LONG_ONLY and side == "short":
            return None          # long-only: shorts were pure drag in research
        pre_score = self.scorer.prerank_score(symbol, side, tech)
        return {"symbol": symbol, "df": df, "tech": tech, "side": side,
                "pre_score": pre_score}

    def _full_evaluate(self, c, equity, committed, held_closes, sentiment, spy_df):
        """Expensive pass on a pre-ranked candidate: quant/regime/MTF/ML -> score
        -> risk gates -> sized entry. Reuses the candidate's df + technical.

        ``committed`` = held positions + symbols with a working order; its length
        is what the position cap is checked against, and a new entry is added to
        it so later candidates in the same scan see the higher count."""
        symbol, df, tech, side = c["symbol"], c["df"], c["tech"], c["side"]

        # ---- Phases 2-6 ------------------------------------------------- #
        quant = self.quant.analyze(df, market_df=spy_df if not spy_df.empty else None)
        regime = self.regime_detector.detect(df, vix=sentiment.vix,
                                             spy_df=spy_df if not spy_df.empty else None)
        mtf = self.mtf.analyze(symbol)
        ml_pred = self._ml_predict(symbol, df)
        research = self.research.analyze(symbol)   # error-safe; neutral if disabled

        # ---- RR plan + Phase 7 score ------------------------------------ #
        sig = Signal(symbol, side, float(df["close"].iloc[-1]),
                     tech.values.get("rsi14") or 50.0, df.index[-1], tech.trend_bias)
        plan = self.rr_filter.evaluate(sig, df)
        score = self.scorer.score(symbol, side, technical=tech, quant=quant,
                                  regime=regime, mtf=mtf, ml=ml_pred, plan=plan,
                                  research=research)
        applied = research.applied_points(side)            # side-aware contribution
        tech_score = score.total - applied                 # base, for reporting

        if research.source_status:
            self._source_status = research.source_status
        self._research_view[symbol] = self._research_card(research)
        dash_row = {"symbol": symbol, "side": side, "score": score.total,
                    "passed": score.passed, "research": applied}
        # High-score watch alert: 80+ that won't actually trade (no plan / gate).
        if score.total >= 80 and (not score.passed or plan is None):
            self.dashboard.alert("high_score", f"{symbol} {side} scored {score.total:.0f}")
            self.notifier.high_score(symbol=symbol, side=side, score=score.total,
                                     reason="no valid 5:1 plan" if plan is None
                                     else f"gated (score {score.total:.0f})")
        if not score.passed or plan is None:
            return dash_row

        # ---- Research vetoes (block long/short/trade) ------------------- #
        ok, why = research.allows(side)
        if not ok:
            logger.info("%s: blocked by research: %s", symbol, why)
            return dash_row

        # ---- Options routing (opt-in): strong signal -> long call/put --- #
        # A qualifying signal buys an ATM option instead of the stock. If no
        # viable/affordable contract exists, fall through to the equity path so
        # the signal isn't wasted.
        if self.options is not None and score.total >= settings.OPTIONS_MIN_SCORE:
            if self._maybe_enter_option(symbol, side, df, score.total, equity):
                return dash_row

        # ---- Phase 8: risk gates + sizing ------------------------------- #
        corr = self.portfolio.max_candidate_correlation(df["close"], held_closes)
        decision = self.portfolio.pre_trade_check(
            score.total, plan.rr, len(committed), candidate_corr=corr,
            current_equity=equity)
        if not decision.allowed:
            logger.info("%s: blocked by risk: %s", symbol, decision.reasons)
            return dash_row

        # Data-only symbols (e.g. crypto Alpaca can't trade) are analyzed but
        # never ordered.
        m = self.meta.get(symbol, {})
        if m.get("asset_class") == "crypto" and m.get("tradable") is False:
            logger.info("%s: data-only (not tradable on Alpaca) — no order", symbol)
            return dash_row

        fraction = self.portfolio.adjust_fraction(decision.risk_fraction,
                                                  regime.volatility, corr)
        if m.get("size_override"):   # e.g. meme coins use a smaller fraction
            fraction = min(fraction, float(m["size_override"]))
        # Research size factor: halve near earnings (<=7d) and/or on a social
        # volume spike (e.g. 1% -> 0.5%, or 0.25% if both).
        if research.size_factor != 1.0:
            logger.info("%s: research size factor %.2f applied", symbol, research.size_factor)
            fraction *= research.size_factor
        sizer = PositionSizer(risk_per_trade=fraction, max_position_pct=settings.MAX_POSITION_PCT)
        trade = sizer.size(plan, equity, fractional=is_crypto(symbol))
        if trade is None:
            return dash_row

        if not self.portfolio.heat_allows(self._open_risk.values(), trade.dollar_risk, equity):
            logger.info("%s: blocked by portfolio heat", symbol)
            return dash_row

        # ---- Phase 9: smart entry --------------------------------------- #
        limit_px = tech.values.get("ema21")  # pullback reference for ranging regimes
        order = self.broker.place_smart_entry(trade, regime_label=regime.label,
                                              limit_price=limit_px)
        if order is not None:
            committed.add(symbol)
            self._open_risk[symbol] = trade.dollar_risk
            logger.info("%s: ENTERED score=%.0f rr=%.1f risk=$%.2f frac=%.3f%%",
                        symbol, score.total, plan.rr, trade.dollar_risk, fraction * 100)
            if settings.RESEARCH_ENABLED:
                self.notifier.trade_opened_research(
                    symbol=symbol, side=side, entry=plan.entry, stop=plan.stop,
                    target=plan.target, rr=plan.rr, tech_score=tech_score,
                    research_bonus=applied, total=score.total,
                    research_lines=research.summary_lines())
            else:
                self.notifier.trade_opened(
                    symbol=symbol, side=side, entry=plan.entry, stop=plan.stop,
                    target=plan.target, rr=plan.rr, score=score.total,
                    dollar_risk=trade.dollar_risk, risk_pct=fraction * 100,
                    regime=regime.label)
            # Register with the stateful position manager (P9).
            self.managed[symbol] = ManagedPosition.from_trade(
                trade, score=score.total, atr=tech.values.get("atr14") or 0.0,
                regime=regime.label, fractional=is_crypto(symbol),
                entry_time=datetime.now(timezone.utc).isoformat())
            self.position_store.save(self.managed)
        return dash_row

    # ------------------------------------------------------------------ #
    # P9 stateful position management
    # ------------------------------------------------------------------ #
    def _manage_positions(self, positions, equity) -> None:
        if not self.managed:
            return
        by_sym = {}
        for p in positions:
            by_sym[p.symbol] = p
            by_sym[p.symbol.replace("/", "")] = p
        for sym in list(self.managed.keys()):
            mp = self.managed[sym]
            pos = by_sym.get(sym) or by_sym.get(sym.replace("/", ""))
            if pos is None:
                if not mp.confirmed:
                    # Pending fill. If there's no longer a working order for it,
                    # the order was canceled/expired -> drop the phantom rather
                    # than tracking it forever. Otherwise keep waiting.
                    if not self.broker.has_open_order(mp.symbol):
                        logger.info("%s: pending order gone — dropping untracked entry",
                                    mp.symbol)
                        self.managed.pop(mp.symbol, None)
                    continue
                # Previously filled, now gone -> its bracket stop/target hit.
                self._finalize_external_close(mp, equity)
                continue
            mp.confirmed = True
            try:
                price = float(getattr(pos, "current_price", mp.entry) or mp.entry)
            except (TypeError, ValueError):
                price = mp.entry
            for action in self.position_manager.update(mp, price, atr=mp.atr):
                self._execute_action(mp, action, price)
            if mp.status == "closed":
                self._finalize_close(mp, equity)
        self.position_store.save(self.managed)

    def _execute_action(self, mp, a, price) -> None:
        d = 1 if mp.side == "long" else -1
        try:
            if a.kind == "scale_out":
                self.broker.scale_out(mp.symbol, mp.side, a.qty)
                self.notifier.scaled_out(symbol=mp.symbol, tag=a.tag, qty=a.qty,
                                         price=a.price, realized_pnl=a.realized_pnl,
                                         remaining=mp.remaining_qty)
            elif a.kind == "move_stop":
                self.broker.replace_stop(mp.symbol, a.new_stop)
                if a.tag == "breakeven":
                    protected = max(0.0, d * (a.new_stop - mp.entry)) * mp.remaining_qty
                    self.notifier.stop_breakeven(symbol=mp.symbol, new_stop=a.new_stop,
                                                 protected_pnl=protected)
                else:
                    profit = d * (price - mp.entry) * mp.remaining_qty
                    self.notifier.trailing_moved(symbol=mp.symbol, new_stop=a.new_stop,
                                                 current_profit=profit)
            elif a.kind == "time_exit":
                self.broker.close_position(mp.symbol)
            # close_hit: the broker bracket already executed the fill.
        except Exception:
            logger.exception("Executing %s action for %s failed", a.kind, mp.symbol)

    def _finalize_close(self, mp, equity) -> None:
        rr = mp.realized_r
        self.notifier.trade_closed(symbol=mp.symbol, side=mp.side, pnl=mp.realized_pnl,
                                   rr_achieved=rr, equity_after=equity)
        self.portfolio.record_trade_result(mp.realized_pnl)
        self._closed_today.append({"symbol": mp.symbol, "pnl": round(mp.realized_pnl, 2),
                                   "r_multiple": round(rr, 2)})
        self._open_risk.pop(mp.symbol, None)
        self.managed.pop(mp.symbol, None)

    def _finalize_external_close(self, mp, equity) -> None:
        """A managed position vanished from the broker — its bracket leg filled.

        We don't get the exact fill, so estimate: assume the target if price was
        within reach of it, else the current (possibly trailed) stop.
        """
        d = 1 if mp.side == "long" else -1
        target_r = d * (mp.target - mp.entry) / mp.risk_per_share if mp.risk_per_share else 0
        close_px = mp.target if mp.last_r >= target_r * 0.9 else mp.current_stop
        mp.realized_pnl += d * (close_px - mp.entry) * mp.remaining_qty
        mp.remaining_qty = 0
        mp.status = "closed"
        logger.info("%s: reconciled external close ~$%.2f (estimated)", mp.symbol, close_px)
        self._finalize_close(mp, equity)

    # ------------------------------------------------------------------ #
    # Options (opt-in) — entry routing + lifecycle management
    # ------------------------------------------------------------------ #
    def _maybe_enter_option(self, symbol, side, df, score, equity) -> bool:
        """Try to express a strong signal as a long call/put. Returns True on entry."""
        if is_crypto(symbol):
            return False
        if len(self.option_positions) >= settings.OPTIONS_MAX_POSITIONS:
            logger.info("%s: option skipped — at max %d option positions",
                        symbol, settings.OPTIONS_MAX_POSITIONS)
            return False
        if any(op.underlying == symbol for op in self.option_positions.values()):
            return False   # already have an option on this underlying

        price = float(df["close"].iloc[-1])
        plan = self.options.plan_trade(symbol, side, price, equity, score)
        if plan is None:
            return False
        q = plan.quote
        order = self.broker.buy_option(q.symbol, plan.contracts)
        if order is None:
            return False

        op = OptionPosition(
            symbol=q.symbol, underlying=symbol, type=q.type, strike=q.strike,
            expiration=q.expiration, contracts=plan.contracts,
            premium_paid=q.premium, cost_basis=plan.cost, side_bias=plan.side_bias,
            score=score, target_premium=plan.target_premium,
            stop_premium=plan.stop_premium,
            entry_time=datetime.now(timezone.utc).isoformat())
        self.option_positions[q.symbol] = op
        self.option_store.save(self.option_positions)
        self._open_risk[q.symbol] = plan.cost   # premium is the defined risk
        logger.info("%s: OPTION ENTERED %s %s x%d @ $%.2f cost=$%.0f score=%.0f — %s",
                    symbol, q.type.upper(), q.symbol, plan.contracts, q.premium,
                    plan.cost, score, plan.description)
        self.notifier.option_bought(
            underlying=symbol, opt_type=q.type, strike=q.strike,
            expiration=q.expiration, contracts=plan.contracts,
            premium_paid=q.premium, cost=plan.cost, description=plan.description)
        return True

    def _manage_options(self, equity) -> None:
        if self.options is None or not self.option_positions:
            return
        broker_opts = {p.symbol: p for p in self.broker.get_option_positions()}
        self._option_live = {}
        for sym in list(self.option_positions.keys()):
            op = self.option_positions[sym]
            bpos = broker_opts.get(sym)
            if bpos is None:
                # Vanished from the broker: settled. If we're at/after expiry it
                # expired worthless; otherwise treat as an external close at $0.
                self._finalize_option(op, current_premium=0.0, equity=equity,
                                      action="expiry", reason="Expired worthless")
                continue
            # Alpaca reports option current_price as the per-share premium.
            try:
                cur = float(getattr(bpos, "current_price", 0) or 0)
            except (TypeError, ValueError):
                cur = 0.0
            if cur <= 0:
                cur = self.options.current_premium(sym) or op.premium_paid
            value = cur * 100 * op.contracts
            pnl = (cur - op.premium_paid) * 100 * op.contracts
            self._option_live[sym] = {"current_premium": cur, "value": value, "pnl": pnl}

            # Milestone: first time up >= 50%.
            if crossed_up50(op, cur):
                self.notifier.option_up(underlying=op.underlying, opt_type=op.type,
                                        strike=op.strike, current_value=value, profit=pnl)
                op.up50_alerted = True

            action, reason = exit_decision(
                op, cur, profit_target=settings.OPTIONS_PROFIT_TARGET,
                stop_loss=settings.OPTIONS_STOP_LOSS,
                expiry_exit_days=settings.OPTIONS_EXPIRY_EXIT_DAYS)
            if action != "hold":
                self.broker.close_option(sym)
                self._finalize_option(op, current_premium=cur, equity=equity,
                                      action=action, reason=reason)
        self.option_store.save(self.option_positions)

    def _finalize_option(self, op, current_premium, equity, action, reason) -> None:
        pnl = (current_premium - op.premium_paid) * 100 * op.contracts
        value = current_premium * 100 * op.contracts
        if action == "take_profit":
            self.notifier.option_doubled(underlying=op.underlying, opt_type=op.type,
                                         strike=op.strike, sold_value=value, profit=pnl)
        elif action == "expiry" and current_premium <= 0:
            self.notifier.option_expired(underlying=op.underlying, opt_type=op.type,
                                         strike=op.strike, loss=op.cost_basis)
        else:
            self.notifier.option_closed(underlying=op.underlying, opt_type=op.type,
                                        strike=op.strike, reason=reason, pnl=pnl)
        self.portfolio.record_trade_result(pnl)
        self._closed_today.append({"symbol": f"{op.underlying} {op.type}",
                                   "pnl": round(pnl, 2),
                                   "r_multiple": round(pnl / op.cost_basis, 2)
                                   if op.cost_basis else 0.0})
        self._open_risk.pop(op.symbol, None)
        self._option_live.pop(op.symbol, None)
        op.status = "closed"
        self.option_positions.pop(op.symbol, None)
        logger.info("%s: OPTION CLOSED (%s) pnl=$%.0f — %s",
                    op.underlying, action, pnl, reason)

    def _ml_predict(self, symbol, df):
        if not settings.ML_ENABLED:
            return None
        ens = self._ml.get(symbol)
        # Train on first encounter, then retrain on the configured cadence.
        retrain_due = self._scan_count > 0 and self._scan_count % (
            settings.ML_RETRAIN_DAYS * max(1, 86400 // settings.SCAN_INTERVAL)) == 0
        if ens is None or retrain_due:
            ens = MLEnsemble()
            if ens.train(df):
                self._ml[symbol] = ens
            else:
                return None
        return ens.predict(df)

    def _sync_held_risk(self, positions, equity) -> None:
        """Make every held broker position count toward the portfolio-heat cap.

        Bot-managed positions already carry their precise entry-time risk in
        ``_open_risk``. For positions not tracked by the manager (e.g. ones that
        predate it), estimate open risk as distance-to-stop x qty when a working
        stop exists, else fall back to the standard per-trade budget (1% of
        equity) — which is how the bot sized them in the first place.
        """
        tracked = set(self.managed.keys())
        tracked |= {op.symbol for op in self.option_positions.values()}
        stops = self.broker.stop_prices()
        for p in positions:
            sym = p.symbol
            if sym in tracked or sym.replace("/", "") in tracked:
                continue   # already counted with its exact risk
            try:
                qty = abs(float(p.qty))
                entry = float(p.avg_entry_price)
            except (TypeError, ValueError):
                continue
            stop = stops.get(sym)
            risk = abs(entry - stop) * qty if stop else settings.RISK_PER_TRADE * equity
            self._open_risk[sym] = round(risk, 2)

    @staticmethod
    def _research_card(r) -> dict:
        """Compact per-symbol research summary for the dashboard."""
        ins, an, nw, so = r.insider, r.analyst, r.news, r.social
        return {
            "points": r.total_points,
            "insider_emoji": ins.emoji if ins else "⚪",
            "insider_summary": ins.summary if ins else "—",
            "analyst_rating": an.rating if an else "N/A",
            "analyst_color": an.badge_color if an else "grey",
            "analyst_n": an.n_analysts if an else 0,
            "target": an.target if an else 0.0,
            "upside_pct": an.upside_pct if an else 0.0,
            "news_emoji": nw.emoji if nw else "⚪",
            "news_headline": nw.top_headline if nw else "",
            "bull_pct": so.bull_pct if so else 0.0,
            "social_status": so.status if so else "error",
            "earnings_label": r.earnings.label,
            "earnings_days": r.earnings.days_to if r.earnings.days_to is not None else -999,
        }

    def _check_earnings_warnings(self, positions) -> None:
        """Telegram-warn once/day for any held position reporting within 1 day."""
        if not settings.RESEARCH_ENABLED:
            return
        today = datetime.now(timezone.utc).date().isoformat()
        for p in positions:
            sym = p.symbol
            if "/" in sym or self._earnings_warned.get(sym) == today:
                continue
            try:
                e = self.research._earnings(sym)
            except Exception:
                continue
            if e.days_to is not None and 0 <= e.days_to <= 1:
                self.notifier.earnings_warning(symbol=sym, days_to=e.days_to)
                self._earnings_warned[sym] = today

    def _held_closes(self, open_symbols):
        closes = {}
        for sym in open_symbols:
            try:
                df = self.feed.get_bars(sym, "1Day", 100)
                if not df.empty:
                    closes[sym] = df["close"]
            except Exception:
                pass
        return closes

    def _build_state(self, positions, scores, equity=0.0, buying_power=0.0, sentiment=None):
        open_pos = []
        for p in positions:
            try:
                open_pos.append({
                    "symbol": p.symbol,
                    "qty": p.qty,
                    "pnl": float(getattr(p, "unrealized_pl", 0) or 0),
                    "pnl_pct": float(getattr(p, "unrealized_plpc", 0) or 0) * 100,
                })
            except Exception:
                continue
        day_start = self.portfolio._day_start_equity or equity
        week_start = self.portfolio._week_start_equity or equity
        return DashboardState(
            equity=equity, buying_power=buying_power,
            daily_pnl=equity - day_start, weekly_pnl=equity - week_start,
            regime_label="(per-symbol)",
            risk_state=sentiment.risk_state if sentiment else "unknown",
            open_positions=open_pos, scores=scores, halted=self.portfolio.halted,
            closed_today=self._closed_today,
        )

    # ------------------------------------------------------------------ #
    # Monitoring integration helpers
    # ------------------------------------------------------------------ #
    def _write_state(self, positions, scores, equity, buying_power, sentiment) -> None:
        st = self._build_state(positions, scores, equity, buying_power, sentiment)
        managed = [{
            "symbol": mp.symbol, "side": mp.side, "entry": mp.entry,
            "current_stop": mp.current_stop, "target": mp.target,
            "remaining_qty": mp.remaining_qty, "realized_pnl": round(mp.realized_pnl, 2),
            "last_r": round(mp.last_r, 2), "bars_held": mp.bars_held,
            "tranches": mp.tranches_taken, "breakeven": mp.breakeven_done,
            "trailing": mp.trailing_active, "score": mp.score, "regime": mp.regime,
        } for mp in self.managed.values()]
        options = []
        for op in self.option_positions.values():
            live = self._option_live.get(op.symbol, {})
            cur = live.get("current_premium", op.premium_paid)
            options.append({
                "symbol": op.symbol, "underlying": op.underlying, "type": op.type,
                "strike": op.strike, "expiration": op.expiration,
                "contracts": op.contracts, "premium_paid": op.premium_paid,
                "cost_basis": op.cost_basis, "current_premium": cur,
                "value": live.get("value", op.cost_basis),
                "pnl": live.get("pnl", 0.0),
                "pnl_pct": ((cur - op.premium_paid) / op.premium_paid * 100)
                if op.premium_paid else 0.0,
                "score": op.score, "side_bias": op.side_bias,
                "description": op.description,
                "target_premium": op.target_premium, "stop_premium": op.stop_premium,
            })
        self.state_store.write_state({
            "equity": st.equity, "buying_power": st.buying_power,
            "daily_pnl": st.daily_pnl, "weekly_pnl": st.weekly_pnl,
            "risk_state": st.risk_state, "halted": st.halted,
            "open_positions": st.open_positions, "scores": st.scores,
            "closed_today": st.closed_today, "managed": managed,
            "options": options,
            "research": self._research_view, "source_status": self._source_status,
        })

    def _detect_closed_trades(self, positions, equity) -> None:
        """Detect positions that closed since last scan and alert (approx PnL).

        Uses the last seen unrealized PnL as an estimate of realized PnL — exact
        realization belongs to the (future) stateful P9 position manager.
        """
        current = {}
        for p in positions:
            try:
                current[p.symbol] = float(getattr(p, "unrealized_pl", 0) or 0)
            except (TypeError, ValueError):
                current[p.symbol] = 0.0
        for sym in set(self._pos_pnl) - set(current):
            if sym in self.managed:
                continue   # exact close handled by the position manager
            pnl = self._pos_pnl.get(sym, 0.0)
            risk = self._open_risk.get(sym) or abs(pnl) or 1.0
            rr = pnl / risk if risk else 0.0
            self.notifier.trade_closed(symbol=sym, side="—", pnl=pnl,
                                       rr_achieved=rr, equity_after=equity)
            self.portfolio.record_trade_result(pnl)
            self._closed_today.append({"symbol": sym, "pnl": round(pnl, 2),
                                       "r_multiple": round(rr, 2)})
            self._open_risk.pop(sym, None)
        self._pos_pnl = current

    def _scheduled_summaries(self, equity, day_start, sentiment) -> None:
        now_et = datetime.now(ZoneInfo("America/New_York"))
        today = now_et.date().isoformat()
        iso = now_et.isocalendar()
        week = f"{iso.year}-W{iso.week}"

        if now_et.hour == 9 and self._sent.get("health") != today:
            self.notifier.system_health(
                regime=sentiment.risk_state if sentiment else "unknown",
                watchlist_n=len(self.watchlist), equity=equity)
            self._sent["health"] = today

        if now_et.hour >= 16 and now_et.weekday() < 5 and self._sent.get("daily") != today:
            self._send_daily_summary(equity, day_start)
            self._sent["daily"] = today
            self._closed_today = []   # reset for the next session/day

        if now_et.weekday() == 4 and now_et.hour >= 16 and self._sent.get("weekly") != week:
            week_start = self.portfolio._week_start_equity or equity
            self.notifier.weekly_summary(
                stats_text=f"Equity ${equity:,.2f} | Week PnL ${equity - week_start:,.2f}")
            self._sent["weekly"] = week

        # Monday ~8am ET: re-screen the S&P 500 and reload the watchlist.
        if now_et.weekday() == 0 and now_et.hour == 8 and self._sent.get("watchlist_week") != week:
            self._refresh_watchlist()
            self._sent["watchlist_week"] = week

    def _refresh_watchlist(self) -> None:
        """Re-run the universe screener (subprocess) and reload the watchlist."""
        import subprocess
        try:
            logger.info("Weekly watchlist refresh: running universe screener…")
            subprocess.run([sys.executable, "scripts/universe_screener.py"],
                           timeout=900, check=False)
            new = settings.load_watchlist()
            if new:
                old = set(self.watchlist)
                self.watchlist = new
                self.meta = settings.load_watchlist_meta()
                logger.info("Watchlist refreshed: %d symbols (+%d/-%d)", len(new),
                            len(set(new) - old), len(old - set(new)))
        except Exception:
            logger.exception("Weekly watchlist refresh failed")
            self.notifier.error_alert("Weekly watchlist refresh failed")

    def _send_daily_summary(self, equity, day_start) -> None:
        ct = self._closed_today
        wins = sum(1 for t in ct if t["pnl"] > 0)
        losses = len(ct) - wins
        best = max(ct, key=lambda t: t["pnl"], default=None)
        worst = min(ct, key=lambda t: t["pnl"], default=None)
        week_start = self.portfolio._week_start_equity or equity
        self.notifier.daily_summary(
            trades=len(ct), wins=wins, losses=losses, pnl=equity - day_start,
            best=f"{best['symbol']} ${best['pnl']:,.2f}" if best else "—",
            worst=f"{worst['symbol']} ${worst['pnl']:,.2f}" if worst else "—",
            equity=equity, weekly=equity - week_start)


def main() -> None:
    configure_logging()
    agent = TradingAgent()
    signal_module.signal(signal_module.SIGINT, agent.shutdown)
    signal_module.signal(signal_module.SIGTERM, agent.shutdown)
    agent.run()


if __name__ == "__main__":
    main()
