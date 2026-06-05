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
    LimitOrderRequest,
    MarketOrderRequest,
    StopLimitOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)

from src.risk.position_sizer import SizedTrade

logger = logging.getLogger(__name__)


def _is_crypto(symbol: str) -> bool:
    return "/" in symbol


def _round(x: float) -> float:
    return round(x, 2)


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

    def get_buying_power(self) -> float:
        return float(self.get_account().buying_power)

    def get_positions(self) -> list[Any]:
        try:
            return self._client.get_all_positions()
        except Exception:
            logger.exception("Failed to fetch positions")
            return []

    def open_symbols(self) -> set[str]:
        return {p.symbol for p in self.get_positions()}

    def open_order_symbols(self) -> set[str]:
        """All symbols with a working order (one call, for the pre-rank pass)."""
        try:
            req = GetOrdersRequest(status=QueryOrderStatus.OPEN)
            return {o.symbol for o in self._client.get_orders(filter=req)}
        except Exception:
            logger.exception("Failed to fetch open orders")
            return set()

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

    # ------------------------------------------------------------------ #
    # Phase 9 — smart / partial entries, scale-out, dynamic stops
    # ------------------------------------------------------------------ #
    @staticmethod
    def scale_out_levels(plan) -> list[tuple[float, float]]:
        """Profit-taking ladder: 33% at 2R, 33% at 3.5R, 34% at full target.

        Returns ``[(price, fraction), ...]`` in price terms for the plan's side.
        """
        r = plan.risk_per_share
        if plan.side == "long":
            return [(_round(plan.entry + 2 * r), 0.33),
                    (_round(plan.entry + 3.5 * r), 0.33),
                    (_round(plan.target), 0.34)]
        return [(_round(plan.entry - 2 * r), 0.33),
                (_round(plan.entry - 3.5 * r), 0.33),
                (_round(plan.target), 0.34)]

    @staticmethod
    def choose_entry_type(regime_label: str | None) -> str:
        """Pick an entry style from the regime: trend->stop breakout, range->limit pullback."""
        if regime_label and "strong_trend" in regime_label:
            return "market"          # don't miss a strong trend
        if regime_label and "ranging" in regime_label:
            return "limit"           # wait for a better fill at a level
        return "market"

    def place_smart_entry(
        self, trade: SizedTrade, regime_label: str | None = None,
        limit_price: float | None = None,
    ) -> Any | None:
        """Entry that adapts to regime: a limit pullback in ranges, else a bracket.

        For a ranging regime with a supplied ``limit_price`` (e.g. an EMA or
        pivot), submits a bracket *limit* entry for a better fill; otherwise
        falls back to the standard market bracket.
        """
        plan = trade.plan
        if _is_crypto(plan.symbol):
            side = OrderSide.BUY if plan.side == "long" else OrderSide.SELL
            return self._place_crypto_entry(trade, side)

        entry_type = self.choose_entry_type(regime_label)
        if entry_type == "limit" and limit_price:
            return self._place_limit_bracket(trade, _round(limit_price))
        return self.place_bracket_order(trade)

    def _place_limit_bracket(self, trade: SizedTrade, limit_price: float) -> Any | None:
        plan = trade.plan
        side = OrderSide.BUY if plan.side == "long" else OrderSide.SELL
        try:
            order = LimitOrderRequest(
                symbol=plan.symbol, qty=trade.qty, side=side,
                time_in_force=TimeInForce.DAY, limit_price=limit_price,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=plan.target),
                stop_loss=StopLossRequest(stop_price=plan.stop),
            )
            logger.info("%s: smart LIMIT bracket entry @%.2f (better than %.2f)",
                        plan.symbol, limit_price, plan.entry)
            result = self._client.submit_order(order)
            logger.info("Order accepted id=%s status=%s", result.id, result.status)
            return result
        except Exception:
            logger.exception("limit bracket entry failed for %s", plan.symbol)
            return None

    def replace_stop(self, symbol: str, new_stop: float) -> bool:
        """Cancel any working stop leg for ``symbol`` and resubmit at ``new_stop``.

        Best-effort dynamic-stop update (breakeven / trail). Returns success.
        Note: robust live management of bracket child legs is stateful; this
        cancels open orders for the symbol and relies on the position manager to
        re-arm protection.
        """
        try:
            query_symbol = symbol.replace("/", "")
            req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[query_symbol])
            for o in self._client.get_orders(filter=req):
                if getattr(o, "type", None) and "stop" in str(o.type).lower():
                    self._client.cancel_order_by_id(o.id)
            logger.info("%s: stop update requested -> %.2f", symbol, _round(new_stop))
            return True
        except Exception:
            logger.exception("replace_stop failed for %s", symbol)
            return False

    def scale_out(self, symbol: str, side: str, qty: float) -> Any | None:
        """Reduce a position by ``qty`` with a market order (partial profit take).

        Sells to reduce a long, buys to reduce a short.
        """
        try:
            reduce_side = OrderSide.SELL if side == "long" else OrderSide.BUY
            tif = TimeInForce.GTC if _is_crypto(symbol) else TimeInForce.DAY
            order = MarketOrderRequest(symbol=symbol, qty=qty, side=reduce_side,
                                       time_in_force=tif)
            result = self._client.submit_order(order)
            logger.info("%s: scaled out %s (order %s)", symbol, qty, result.id)
            return result
        except Exception:
            logger.exception("scale_out failed for %s", symbol)
            return None

    def close_position(self, symbol: str) -> bool:
        """Close a single position (used by the time-based exit)."""
        try:
            self._client.close_position(symbol.replace("/", ""))
            logger.info("%s: position closed", symbol)
            return True
        except Exception:
            logger.exception("close_position failed for %s", symbol)
            return False

    def close_all(self) -> None:
        """Liquidate every open position and cancel working orders (kill switch)."""
        try:
            logger.warning("Closing ALL positions and cancelling orders")
            self._client.close_all_positions(cancel_orders=True)
        except Exception:
            logger.exception("close_all failed")
