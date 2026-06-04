"""Live trading agent entry point.

Scans the watchlist on a fixed interval and, for each symbol, runs the full
decision pipeline:

    multi-timeframe alignment  ->  EMA/RSI signal  ->  5:1 RR filter
        ->  portfolio-risk gates  ->  position sizing  ->  bracket order

A daily-loss kill switch flattens the book and halts trading. Ctrl+C triggers a
graceful shutdown. Everything is logged to ``logs/agent_YYYYMMDD.log``.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone

from config import settings
from src.data.feed import MarketFeed, is_crypto
from src.execution.broker import Broker
from src.risk.portfolio_risk import PortfolioRisk
from src.risk.position_sizer import PositionSizer
from src.signals.multi_timeframe import MultiTimeframe
from src.signals.rr_filter import RRFilter
from src.signals.strategy import EMAStrategy

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
    logger.info("Logging to %s", log_path)


class TradingAgent:
    def __init__(self) -> None:
        self._validate_credentials()
        self.feed = MarketFeed(
            settings.ALPACA_API_KEY,
            settings.ALPACA_SECRET_KEY,
            stock_feed=settings.STOCK_DATA_FEED,
        )
        self.strategy = EMAStrategy(
            fast=settings.EMA_FAST,
            slow=settings.EMA_SLOW,
            rsi_period=settings.RSI_PERIOD,
            rsi_long_threshold=settings.RSI_LONG_THRESHOLD,
            rsi_short_threshold=settings.RSI_SHORT_THRESHOLD,
        )
        self.mtf = MultiTimeframe(
            self.feed, self.strategy,
            timeframes=settings.MTF_TIMEFRAMES,
            lookback=settings.LOOKBACK_BARS,
        )
        self.rr_filter = RRFilter(
            rr_ratio=settings.RR_RATIO,
            atr_period=settings.ATR_PERIOD,
            atr_multiplier=settings.ATR_MULTIPLIER,
            swing_lookback=settings.SWING_LOOKBACK,
        )
        self.sizer = PositionSizer(
            risk_per_trade=settings.RISK_PER_TRADE,
            max_position_pct=settings.MAX_POSITION_PCT,
        )
        self.portfolio = PortfolioRisk(
            max_positions=settings.MAX_CONCURRENT_POSITIONS,
            daily_loss_limit=settings.DAILY_LOSS_LIMIT,
            max_correlation=settings.MAX_CORRELATION,
        )
        self.broker = Broker(
            settings.ALPACA_API_KEY,
            settings.ALPACA_SECRET_KEY,
            paper=settings.PAPER,
        )
        self._running = True
        self._halted = False

    @staticmethod
    def _validate_credentials() -> None:
        if not settings.ALPACA_API_KEY or not settings.ALPACA_SECRET_KEY:
            sys.exit(
                "Missing ALPACA_API_KEY / ALPACA_SECRET_KEY in .env "
                "(paper-trading keys required)."
            )

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def shutdown(self, *_args) -> None:
        logger.info("Shutdown requested — finishing current cycle and exiting.")
        self._running = False

    def run(self) -> None:
        # Establish the kill-switch baseline from current equity.
        try:
            self.portfolio.set_day_start_equity(self.broker.get_equity())
        except Exception:
            logger.exception("Could not read starting equity; kill switch disabled until next read")

        logger.info(
            "Agent started | mode=%s | watchlist=%s | interval=%ds",
            "PAPER" if settings.PAPER else "LIVE", settings.WATCHLIST, settings.SCAN_INTERVAL,
        )
        while self._running:
            try:
                self.scan_once()
            except Exception:
                logger.exception("Scan cycle failed")
            if not self._running:
                break
            self._sleep(settings.SCAN_INTERVAL)
        logger.info("Agent stopped.")

    def _sleep(self, seconds: int) -> None:
        """Sleep in 1s steps so Ctrl+C is responsive between scans."""
        for _ in range(seconds):
            if not self._running:
                return
            time.sleep(1)

    # ------------------------------------------------------------------ #
    # One scan over the watchlist
    # ------------------------------------------------------------------ #
    def scan_once(self) -> None:
        if self._halted:
            logger.warning("Trading halted by kill switch — no new orders.")
            return

        equity = self.broker.get_equity()

        # Kill switch first: liquidate and stop trading for the session.
        if self.portfolio.kill_switch_triggered(equity):
            self.broker.close_all()
            self._halted = True
            return

        positions = self.broker.get_positions()
        open_symbols = {p.symbol for p in positions}
        held_closes = self._held_closes(open_symbols)

        logger.info(
            "Scan | equity=$%.2f open=%d/%d exposure=%.0f%%",
            equity, len(positions), settings.MAX_CONCURRENT_POSITIONS,
            self.portfolio.gross_exposure(positions, equity) * 100,
        )

        for symbol in settings.WATCHLIST:
            try:
                self._evaluate_symbol(symbol, equity, open_symbols, held_closes)
            except Exception:
                logger.exception("Error evaluating %s", symbol)

    def _evaluate_symbol(self, symbol, equity, open_symbols, held_closes) -> None:
        # Alpaca reports crypto positions without the slash (BTCUSD).
        if symbol in open_symbols or symbol.replace("/", "") in open_symbols:
            logger.info("%s: already in a position, skipping", symbol)
            return
        if not self.portfolio.can_open_new(len(open_symbols)):
            return
        if self.broker.has_open_order(symbol):
            logger.info("%s: working order exists, skipping", symbol)
            return

        # 1) Multi-timeframe trend agreement.
        direction = self.mtf.aligned_direction(symbol)
        if direction is None:
            return

        # 2) Entry-timeframe EMA/RSI signal must match the aligned direction.
        df = self.feed.get_bars(symbol, settings.SIGNAL_TIMEFRAME, settings.LOOKBACK_BARS)
        if df.empty:
            return
        signal = self.strategy.evaluate(symbol, df)
        if signal is None or signal.side != direction:
            logger.info("%s: no aligned signal (mtf=%s)", symbol, direction)
            return
        logger.info("%s: %s signal — %s", symbol, signal.side, signal.reason)

        # 3) 5:1 reward/risk filter with ATR stop + structural target.
        plan = self.rr_filter.evaluate(signal, df)
        if plan is None:
            return

        # 4) Portfolio correlation gate.
        if not self.portfolio.correlation_ok(df["close"], held_closes):
            logger.info("%s: blocked by correlation gate", symbol)
            return

        # 5) Position sizing (1% risk, 10% cap).
        trade = self.sizer.size(plan, equity, fractional=is_crypto(symbol))
        if trade is None:
            return

        # 6) Submit the bracket order.
        order = self.broker.place_bracket_order(trade)
        if order is not None:
            open_symbols.add(symbol)  # count it toward the cap this cycle

    def _held_closes(self, open_symbols: set[str]) -> dict:
        """Recent closes for each held symbol, for the correlation check."""
        closes: dict = {}
        for sym in open_symbols:
            try:
                df = self.feed.get_bars(sym, settings.SIGNAL_TIMEFRAME, 100)
                if not df.empty:
                    closes[sym] = df["close"]
            except Exception:
                logger.debug("Could not load closes for held %s", sym)
        return closes


def main() -> None:
    configure_logging()
    agent = TradingAgent()
    # Graceful shutdown on Ctrl+C / SIGTERM.
    signal.signal(signal.SIGINT, agent.shutdown)
    signal.signal(signal.SIGTERM, agent.shutdown)
    agent.run()


if __name__ == "__main__":
    main()
