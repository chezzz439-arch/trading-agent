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

from config import settings
from src.data.feed import MarketFeed, is_crypto
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
        self.scorer = MasterScorer(min_score=settings.MIN_SCORE)
        self.rr_filter = RRFilter(rr_ratio=settings.RR_RATIO, atr_period=settings.ATR_PERIOD,
                                  atr_multiplier=settings.ATR_MULTIPLIER,
                                  swing_lookback=settings.SWING_LOOKBACK)
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
        self._ml: dict[str, MLEnsemble] = {}     # per-symbol ensemble cache
        self._running = True
        self._open_risk: dict[str, float] = {}   # symbol -> intended dollar risk
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
        try:
            eq = self.broker.get_equity()
            self.portfolio.set_day_start_equity(eq)
            self.portfolio.set_week_start_equity(eq)
        except Exception:
            logger.exception("Could not read starting equity")
        logger.info("Agent started | mode=%s | %d symbols | interval=%ds | min_score=%.0f",
                    "PAPER" if settings.PAPER else "LIVE", len(settings.WATCHLIST),
                    settings.SCAN_INTERVAL, settings.MIN_SCORE)
        while self._running:
            try:
                self.scan_once()
            except Exception:
                logger.exception("Scan cycle failed")
            self._scan_count += 1
            if not self._running:
                break
            self._sleep(settings.SCAN_INTERVAL)
        self._on_exit()

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
        logger.info("Agent stopped.")

    # ------------------------------------------------------------------ #
    def scan_once(self) -> None:
        equity = self.broker.get_equity()
        buying_power = self.broker.get_buying_power()

        if self.portfolio.kill_switch_triggered(equity):
            self.broker.close_all()
            self.dashboard.alert("kill_switch", "Trading halted — book flattened")
            return

        sentiment = self.sentiment.analyze(now=time.time())
        spy_df = self.feed.get_bars(settings.MARKET_PROXY, "1Day", settings.LOOKBACK_BARS)

        positions = self.broker.get_positions()
        open_symbols = {p.symbol for p in positions}
        held_closes = self._held_closes(open_symbols)

        scores_for_dash = []
        for symbol in settings.WATCHLIST:
            try:
                result = self._evaluate(symbol, equity, open_symbols, held_closes,
                                        sentiment, spy_df)
                if result is not None:
                    scores_for_dash.append(result)
            except Exception:
                logger.exception("Error evaluating %s", symbol)

        state = self._build_state(positions, scores_for_dash, equity, buying_power,
                                  sentiment)
        self.dashboard.print(state)

    def _evaluate(self, symbol, equity, open_symbols, held_closes, sentiment, spy_df):
        if symbol in open_symbols or symbol.replace("/", "") in open_symbols:
            return None
        if self.broker.has_open_order(symbol):
            return None

        df = self.feed.get_bars(symbol, "1Day", settings.LOOKBACK_BARS)
        if df.empty or len(df) < 60:
            return None

        # ---- Phase 1: technical -> candidate direction ------------------ #
        tech = self.technical.analyze(df)
        if tech is None or tech.trend_bias == "neutral":
            return None
        side = tech.trend_bias

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
        if score.total >= 80:
            self.dashboard.alert("high_score", f"{symbol} {side} scored {score.total:.0f}")
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
        return dash_row

    # ------------------------------------------------------------------ #
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
        )


def main() -> None:
    configure_logging()
    agent = TradingAgent()
    signal_module.signal(signal_module.SIGINT, agent.shutdown)
    signal_module.signal(signal_module.SIGTERM, agent.shutdown)
    agent.run()


if __name__ == "__main__":
    main()
