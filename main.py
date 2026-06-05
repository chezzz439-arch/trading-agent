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
        self.feed = MarketFeed(k, s, stock_feed=settings.STOCK_DATA_FEED)
        self.technical = TechnicalAnalysis()
        self.quant = QuantAnalysis()
        self.regime_detector = RegimeDetector()
        self.sentiment = SentimentAnalyzer(ttl=settings.SCAN_INTERVAL)
        self.mtf = MTFConfluence(self.feed, timeframes=settings.MTF_TIMEFRAMES,
                                 lookback=settings.LOOKBACK_BARS,
                                 min_confluence=settings.MIN_CONFLUENCE)
        self.scorer = MasterScorer(min_score=settings.MIN_SCORE, rr_target=settings.RR_RATIO)
        self.rr_filter = RRFilter(rr_ratio=settings.RR_RATIO, atr_period=settings.ATR_PERIOD,
                                  atr_multiplier=settings.ATR_MULTIPLIER,
                                  swing_lookback=settings.SWING_LOOKBACK,
                                  path_veto=settings.RR_PATH_VETO)
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
        # Dynamic watchlist (screened) + per-symbol eligibility gate.
        self.watchlist = settings.load_watchlist()
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
        self._ml: dict[str, MLEnsemble] = {}     # per-symbol ensemble cache
        self._running = True
        self._open_risk: dict[str, float] = {}   # symbol -> intended dollar risk
        self._pos_pnl: dict[str, float] = {}     # last seen unrealized PnL per symbol
        self._closed_today: list[dict] = []
        self._sent: dict[str, str] = {}          # summary-type -> date/week already sent
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
        held_closes = self._held_closes(open_symbols)

        order_syms = self.broker.open_order_symbols()

        # --- Pre-rank: cheap technical-only pass over the whole universe -- #
        candidates = []
        for symbol in self.watchlist:
            try:
                c = self._prerank(symbol, open_symbols, order_syms)
                if c is not None:
                    candidates.append(c)
            except Exception:
                logger.exception("Pre-rank failed for %s", symbol)
        candidates.sort(key=lambda c: c["pre_score"], reverse=True)
        top = candidates[: settings.PRERANK_TOP_N]
        logger.info("Pre-ranked %d/%d candidates; deep-analyzing top %d",
                    len(candidates), len(self.watchlist), len(top))

        # --- Full pass: expensive pipeline (quant/MTF/ML) only on top N --- #
        scores_for_dash = []
        for c in top:
            try:
                row = self._full_evaluate(c, equity, open_symbols, held_closes,
                                          sentiment, spy_df)
                if row is not None:
                    scores_for_dash.append(row)
            except Exception:
                logger.exception("Error evaluating %s", c["symbol"])

        state = self._build_state(positions, scores_for_dash, equity, buying_power, sentiment)
        self.dashboard.print(state)
        self._write_state(positions, scores_for_dash, equity, buying_power, sentiment)
        self._scheduled_summaries(equity, day_start, sentiment)

    def _prerank(self, symbol, open_symbols, order_syms):
        """Cheap pass: bars + screen + technical -> a fast ranking score.

        Returns a candidate dict (reusing the fetched df + technical so the full
        pass doesn't recompute them) or None to skip the symbol.
        """
        if symbol in open_symbols or symbol.replace("/", "") in open_symbols:
            return None
        if symbol in order_syms or symbol.replace("/", "") in order_syms:
            return None

        df = self.feed.get_bars(symbol, "1Day", settings.LOOKBACK_BARS)
        if df.empty or len(df) < 60:
            return None

        eligible, reason = self.screener.passes(symbol, df)
        if not eligible:
            logger.info("%s: screened out — %s", symbol, reason)
            return None

        tech = self.technical.analyze(df)
        if tech is None or tech.trend_bias == "neutral":
            return None
        side = tech.trend_bias
        pre_score = self.scorer.prerank_score(symbol, side, tech)
        return {"symbol": symbol, "df": df, "tech": tech, "side": side,
                "pre_score": pre_score}

    def _full_evaluate(self, c, equity, open_symbols, held_closes, sentiment, spy_df):
        """Expensive pass on a pre-ranked candidate: quant/regime/MTF/ML -> score
        -> risk gates -> sized entry. Reuses the candidate's df + technical."""
        symbol, df, tech, side = c["symbol"], c["df"], c["tech"], c["side"]

        # ---- Phases 2-6 ------------------------------------------------- #
        quant = self.quant.analyze(df, market_df=spy_df if not spy_df.empty else None)
        regime = self.regime_detector.detect(df, vix=sentiment.vix,
                                             spy_df=spy_df if not spy_df.empty else None)
        mtf = self.mtf.analyze(symbol)
        ml_pred = self._ml_predict(symbol, df)

        # ---- RR plan + Phase 7 score ------------------------------------ #
        sig = Signal(symbol, side, float(df["close"].iloc[-1]),
                     tech.values.get("rsi14") or 50.0, df.index[-1], tech.trend_bias)
        plan = self.rr_filter.evaluate(sig, df)
        score = self.scorer.score(symbol, side, technical=tech, quant=quant,
                                  regime=regime, mtf=mtf, ml=ml_pred, plan=plan)

        dash_row = {"symbol": symbol, "side": side, "score": score.total,
                    "passed": score.passed}
        # High-score watch alert: 80+ that won't actually trade (no plan / gate).
        if score.total >= 80 and (not score.passed or plan is None):
            self.dashboard.alert("high_score", f"{symbol} {side} scored {score.total:.0f}")
            self.notifier.high_score(symbol=symbol, side=side, score=score.total,
                                     reason="no valid 5:1 plan" if plan is None
                                     else f"gated (score {score.total:.0f})")
        if not score.passed or plan is None:
            return dash_row

        # ---- Phase 8: risk gates + sizing ------------------------------- #
        corr = self.portfolio.max_candidate_correlation(df["close"], held_closes)
        decision = self.portfolio.pre_trade_check(
            score.total, plan.rr, len(open_symbols), candidate_corr=corr,
            current_equity=equity)
        if not decision.allowed:
            logger.info("%s: blocked by risk: %s", symbol, decision.reasons)
            return dash_row

        fraction = self.portfolio.adjust_fraction(decision.risk_fraction,
                                                  regime.volatility, corr)
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
            open_symbols.add(symbol)
            self._open_risk[symbol] = trade.dollar_risk
            logger.info("%s: ENTERED score=%.0f rr=%.1f risk=$%.2f frac=%.3f%%",
                        symbol, score.total, plan.rr, trade.dollar_risk, fraction * 100)
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
        self.state_store.write_state({
            "equity": st.equity, "buying_power": st.buying_power,
            "daily_pnl": st.daily_pnl, "weekly_pnl": st.weekly_pnl,
            "risk_state": st.risk_state, "halted": st.halted,
            "open_positions": st.open_positions, "scores": st.scores,
            "closed_today": st.closed_today, "managed": managed,
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
