"""Alpaca execution layer.

``Broker`` wraps ``TradingClient`` and exposes account state plus atomic
bracket-order submission (entry + stop-loss + take-profit).

Note on crypto: Alpaca's crypto venue does **not** support bracket orders or
short selling. For crypto symbols the broker degrades gracefully — it submits a
simple market entry and logs a warning that the protective legs must be managed
separately, and refuses short crypto orders.
"""

from __future__ import annotations

import logging
from typing import Any

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import (
    OrderClass,
    OrderSide,
    QueryOrderStatus,
    TimeInForce,
)
from alpaca.trading.requests import (
    GetOrdersRequest,
    MarketOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)

from src.risk.position_sizer import SizedTrade

logger = logging.getLogger(__name__)


def _is_crypto(symbol: str) -> bool:
    return "/" in symbol


class Broker:
    def __init__(self, api_key: str, secret_key: str, paper: bool = True) -> None:
        self._client = TradingClient(api_key, secret_key, paper=paper)
        self.paper = paper

    # ------------------------------------------------------------------ #
    # Account / positions
    # ------------------------------------------------------------------ #
    def get_account(self) -> Any:
        return self._client.get_account()

    def get_equity(self) -> float:
        return float(self.get_account().equity)

    def get_positions(self) -> list[Any]:
        try:
            return self._client.get_all_positions()
        except Exception:
            logger.exception("Failed to fetch positions")
            return []

    def open_symbols(self) -> set[str]:
        return {p.symbol for p in self.get_positions()}

    def has_open_order(self, symbol: str) -> bool:
        """True if a working order already exists for ``symbol``."""
        try:
            # Alpaca order symbols for crypto drop the slash (BTC/USD -> BTCUSD).
            query_symbol = symbol.replace("/", "")
            req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[query_symbol])
            return len(self._client.get_orders(filter=req)) > 0
        except Exception:
            logger.exception("Failed to check open orders for %s", symbol)
            return False

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #
    def place_bracket_order(self, trade: SizedTrade) -> Any | None:
        """Submit an atomic bracket order (entry + stop + take-profit).

        Returns the order object on success, or ``None`` on failure / when the
        order is rejected for venue limitations (e.g. short crypto).
        """
        plan = trade.plan
        side = OrderSide.BUY if plan.side == "long" else OrderSide.SELL

        try:
            if _is_crypto(plan.symbol):
                return self._place_crypto_entry(trade, side)

            order = MarketOrderRequest(
                symbol=plan.symbol,
                qty=trade.qty,
                side=side,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=plan.target),
                stop_loss=StopLossRequest(stop_price=plan.stop),
            )
            logger.info(
                "Submitting %s BRACKET %s qty=%s entry~%.2f stop=%.2f tp=%.2f "
                "(risk $%.2f, RR %.1f)",
                plan.side.upper(), plan.symbol, trade.qty, plan.entry,
                plan.stop, plan.target, trade.dollar_risk, plan.rr,
            )
            result = self._client.submit_order(order)
            logger.info("Order accepted id=%s status=%s", result.id, result.status)
            return result
        except Exception:
            logger.exception("place_bracket_order failed for %s", plan.symbol)
            return None

    def _place_crypto_entry(self, trade: SizedTrade, side: OrderSide) -> Any | None:
        plan = trade.plan
        if side == OrderSide.SELL:
            logger.warning(
                "%s: short crypto is not supported on Alpaca — skipping", plan.symbol
            )
            return None
        logger.warning(
            "%s: crypto does not support bracket orders — submitting simple "
            "market entry; manage stop=%.2f / target=%.2f separately",
            plan.symbol, plan.stop, plan.target,
        )
        order = MarketOrderRequest(
            symbol=plan.symbol,
            qty=trade.qty,
            side=side,
            time_in_force=TimeInForce.GTC,
        )
        result = self._client.submit_order(order)
        logger.info("Crypto entry accepted id=%s status=%s", result.id, result.status)
        return result

    def close_all(self) -> None:
        """Liquidate every open position and cancel working orders (kill switch)."""
        try:
            logger.warning("Closing ALL positions and cancelling orders")
            self._client.close_all_positions(cancel_orders=True)
        except Exception:
            logger.exception("close_all failed")
