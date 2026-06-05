"""Layer 2 — Telegram alerts.

``TelegramNotifier`` sends formatted alerts via the Telegram Bot HTTP API using
``requests`` (synchronous, robust inside the agent's scan loop — no async event
loop to manage). Credentials come from ``TELEGRAM_BOT_TOKEN`` /
``TELEGRAM_CHAT_ID``; if either is missing the notifier is disabled and every
call is a silent no-op, so the agent runs fine without Telegram configured.

Messages use legacy Markdown (``*bold*``). If a send fails to parse, it is
retried as plain text so an alert is never lost to a formatting issue.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_API = "https://api.telegram.org/bot{token}/{method}"


class TelegramNotifier:
    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None):
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self.enabled = bool(self.token and self.chat_id)
        if not self.enabled:
            logger.info("Telegram disabled (no token/chat id) — alerts will no-op")

    # ------------------------------------------------------------------ #
    # Transport
    # ------------------------------------------------------------------ #
    def _post(self, method: str, payload: dict, files=None) -> bool:
        if not self.enabled:
            return False
        try:
            url = _API.format(token=self.token, method=method)
            r = requests.post(url, data=payload, files=files, timeout=15)
            ok = r.json().get("ok", False)
            if not ok:
                logger.warning("Telegram %s failed: %s", method, r.json().get("description"))
            return bool(ok)
        except Exception:
            logger.exception("Telegram %s request failed", method)
            return False

    def send(self, text: str) -> bool:
        payload = {"chat_id": self.chat_id, "text": text, "parse_mode": "Markdown"}
        if self._post("sendMessage", payload):
            return True
        # Retry without markdown if parsing failed.
        return self._post("sendMessage", {"chat_id": self.chat_id, "text": text})

    def send_photo(self, image_path: str, caption: str = "") -> bool:
        if not self.enabled or not os.path.exists(image_path):
            return False
        try:
            with open(image_path, "rb") as img:
                return self._post("sendPhoto",
                                  {"chat_id": self.chat_id, "caption": caption},
                                  files={"photo": img})
        except Exception:
            logger.exception("Telegram send_photo failed")
            return False

    def test(self) -> bool:
        return self.send("*Trading Agent* — Telegram connection OK ✅")

    # ------------------------------------------------------------------ #
    # Typed alerts (the 10 required notifications)
    # ------------------------------------------------------------------ #
    def startup(self, mode: str, equity: float) -> bool:
        return self.send(f"*Trading Agent Started* 🚀\nMode: {mode}\n"
                         f"Equity: ${equity:,.2f}")

    def trade_opened(self, *, symbol, side, entry, stop, target, rr, score,
                     dollar_risk, risk_pct, regime) -> bool:
        emoji = "🟢" if side == "long" else "🔴"
        stop_pct = (stop - entry) / entry * 100
        tgt_pct = (target - entry) / entry * 100
        return self.send(
            f"*TRADE OPENED* {emoji}\n"
            f"Symbol: {symbol} {side.upper()}\n"
            f"Entry: ${entry:,.2f}\n"
            f"Stop: ${stop:,.2f} ({stop_pct:+.2f}%)\n"
            f"Target: ${target:,.2f} ({tgt_pct:+.2f}%)\n"
            f"RR: {rr:.1f}:1\n"
            f"Score: {score:.0f}/100\n"
            f"Risk: ${dollar_risk:,.2f} ({risk_pct:.2f}% account)\n"
            f"Regime: {regime}")

    def trade_closed(self, *, symbol, side, pnl, rr_achieved, equity_after) -> bool:
        win = pnl >= 0
        return self.send(
            f"*TRADE CLOSED* {'✅' if win else '❌'}\n"
            f"Symbol: {symbol} {side.upper()}\n"
            f"PnL: ${pnl:,.2f} ({'WIN' if win else 'LOSS'})\n"
            f"RR achieved: {rr_achieved:+.2f}\n"
            f"Equity after: ${equity_after:,.2f}")

    def stop_breakeven(self, *, symbol, new_stop, protected_pnl) -> bool:
        return self.send(
            f"*STOP → BREAKEVEN* 🛡️\nSymbol: {symbol}\n"
            f"New stop: ${new_stop:,.2f}\nUnrealized protected: ${protected_pnl:,.2f}")

    def trailing_moved(self, *, symbol, new_stop, current_profit) -> bool:
        return self.send(
            f"*TRAILING STOP MOVED* 📈\nSymbol: {symbol}\n"
            f"New trailing stop: ${new_stop:,.2f}\nCurrent profit: ${current_profit:,.2f}")

    def kill_switch(self, *, reason, daily_loss) -> bool:
        return self.send(
            f"*KILL SWITCH TRIGGERED* 🚨\nReason: {reason}\n"
            f"Daily loss: ${daily_loss:,.2f}\nAll positions closed.")

    def high_score(self, *, symbol, side, score, reason) -> bool:
        return self.send(
            f"*HIGH SCORE SIGNAL* 👀\n{symbol} {side.upper()} scored {score:.0f}/100\n"
            f"(did not meet RR/gate — watch manually)\n{reason}")

    def daily_summary(self, *, trades, wins, losses, pnl, best, worst, equity, weekly) -> bool:
        return self.send(
            f"*DAILY SUMMARY* 📊\n"
            f"Trades: {trades} (W {wins} / L {losses})\n"
            f"Daily PnL: ${pnl:,.2f}\n"
            f"Best: {best}\nWorst: {worst}\n"
            f"Equity: ${equity:,.2f}\nWeek running: ${weekly:,.2f}")

    def weekly_summary(self, *, stats_text, image_path=None) -> bool:
        if image_path:
            return self.send_photo(image_path, caption=f"*WEEKLY SUMMARY* 🗓️\n{stats_text}")
        return self.send(f"*WEEKLY SUMMARY* 🗓️\n{stats_text}")

    def system_health(self, *, regime, watchlist_n, equity) -> bool:
        return self.send(
            f"*SYSTEM HEALTH* 💚\nAgent running ✓\nRegime: {regime}\n"
            f"Watchlist: {watchlist_n} symbols\nEquity: ${equity:,.2f}")

    def error_alert(self, message: str) -> bool:
        return self.send(f"*ERROR ALERT* ⚠️\n{message}")
