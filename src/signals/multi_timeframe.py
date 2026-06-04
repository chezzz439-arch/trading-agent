"""Multi-timeframe trend alignment.

A trade is only allowed when the directional bias agrees across two
timeframes (default 1h and 4h). This filters counter-trend crossovers on the
entry timeframe that fight the higher-timeframe trend.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.data.feed import MarketFeed
from src.signals.strategy import EMAStrategy

logger = logging.getLogger(__name__)


class MultiTimeframe:
    def __init__(
        self,
        feed: MarketFeed,
        strategy: EMAStrategy,
        timeframes: tuple[str, str] = ("1Hour", "4Hour"),
        lookback: int = 300,
    ) -> None:
        self.feed = feed
        self.strategy = strategy
        self.timeframes = timeframes
        self.lookback = lookback

    def aligned_direction(self, symbol: str) -> Optional[str]:
        """Return 'long'/'short' if all timeframes agree, else None.

        Never raises: any data/compute error for a timeframe is treated as
        "no agreement" so the caller simply skips the symbol this cycle.
        """
        biases = []
        for tf in self.timeframes:
            try:
                df = self.feed.get_bars(symbol, tf, self.lookback)
                bias = self.strategy.bias(df)
            except Exception:
                logger.exception("MTF bias failed for %s @ %s", symbol, tf)
                bias = None
            if bias is None:
                logger.debug("%s @ %s: no bias", symbol, tf)
                return None
            biases.append(bias)

        if all(b == biases[0] for b in biases):
            logger.info("%s: timeframes aligned -> %s", symbol, biases[0])
            return biases[0]
        logger.debug("%s: timeframes disagree %s", symbol, biases)
        return None
