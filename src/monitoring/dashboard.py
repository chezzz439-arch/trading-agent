"""Phase 10 — Monitoring: console dashboard, alerts, and daily report.

``Dashboard`` is a pure presentation layer: it takes plain state (account,
positions, scores, regime, PnL, recent trades) and renders a console snapshot,
prints alerts, and writes an end-of-day report. It holds no trading logic.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

_BAR = "=" * 78


@dataclass
class DashboardState:
    equity: float = 0.0
    buying_power: float = 0.0
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    regime_label: str = "unknown"
    risk_state: str = "unknown"
    open_positions: list = field(default_factory=list)   # dicts: symbol, qty, pnl, pnl_pct
    closed_today: list = field(default_factory=list)      # dicts: symbol, pnl, r_multiple
    scores: list = field(default_factory=list)            # dicts: symbol, side, score, passed
    win_rate_20: Optional[float] = None
    halted: bool = False


class Dashboard:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir

    # ------------------------------------------------------------------ #
    def render(self, st: DashboardState, now: Optional[datetime] = None) -> str:
        now = now or datetime.now(timezone.utc)
        lines = [
            _BAR,
            f" TRADING AGENT  {now:%Y-%m-%d %H:%M:%S} UTC"
            f"{'   *** HALTED ***' if st.halted else ''}",
            _BAR,
            f" Equity ${st.equity:,.2f}   BuyingPower ${st.buying_power:,.2f}   "
            f"Daily {self._sign(st.daily_pnl)}   Weekly {self._sign(st.weekly_pnl)}",
            f" Regime: {st.regime_label}   Market: {st.risk_state}   "
            f"WinRate(20): {self._pct(st.win_rate_20)}",
            "-" * 78,
            f" OPEN POSITIONS ({len(st.open_positions)})",
        ]
        if st.open_positions:
            for p in st.open_positions:
                lines.append(
                    f"   {p['symbol']:<10} qty {p.get('qty',0):<10} "
                    f"PnL {self._sign(p.get('pnl',0))} ({p.get('pnl_pct',0):+.2f}%)"
                )
        else:
            lines.append("   (none)")

        lines += ["-" * 78, " SIGNAL SCORES"]
        if st.scores:
            for s in sorted(st.scores, key=lambda d: d["score"], reverse=True):
                flag = "TRADE" if s["passed"] else "  -  "
                lines.append(
                    f"   {s['symbol']:<10} {s['side']:<6} {s['score']:>5.1f}/100  [{flag}]"
                )
        else:
            lines.append("   (no candidates this scan)")

        lines += ["-" * 78, f" CLOSED TODAY ({len(st.closed_today)})"]
        if st.closed_today:
            for t in st.closed_today:
                lines.append(
                    f"   {t['symbol']:<10} PnL {self._sign(t.get('pnl',0))}  "
                    f"{t.get('r_multiple',0):+.2f}R"
                )
        else:
            lines.append("   (none)")
        lines.append(_BAR)
        return "\n".join(lines)

    def print(self, st: DashboardState) -> None:
        print(self.render(st))

    # ------------------------------------------------------------------ #
    def alert(self, kind: str, message: str) -> None:
        """Emit a prominent alert (kill switch, high score, 2R hit, stop moved)."""
        banner = f"  ALERT [{kind.upper()}]  {message}"
        line = "!" * min(78, len(banner) + 4)
        print(f"\n{line}\n{banner}\n{line}")
        logger.warning("ALERT %s: %s", kind, message)

    # ------------------------------------------------------------------ #
    def daily_report(self, st: DashboardState, now: Optional[datetime] = None) -> str:
        """Write an end-of-day summary to logs/daily_report_YYYYMMDD.txt."""
        now = now or datetime.now(timezone.utc)
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, f"daily_report_{now:%Y%m%d}.txt")
        wins = sum(1 for t in st.closed_today if t.get("pnl", 0) > 0)
        total = len(st.closed_today)
        avg_r = (sum(t.get("r_multiple", 0) for t in st.closed_today) / total) if total else 0.0
        body = [
            _BAR,
            f" DAILY REPORT — {now:%Y-%m-%d}",
            _BAR,
            f" Closing equity : ${st.equity:,.2f}",
            f" Daily PnL      : {self._sign(st.daily_pnl)}",
            f" Weekly PnL     : {self._sign(st.weekly_pnl)}",
            f" Regime         : {st.regime_label}",
            f" Trades closed  : {total}  (wins {wins}, win rate "
            f"{(wins/total*100 if total else 0):.0f}%)",
            f" Avg R-multiple : {avg_r:+.2f}R",
            f" Open positions : {len(st.open_positions)}",
            f" Halted         : {st.halted}",
            _BAR,
            "",
            " Trade log:",
        ]
        for t in st.closed_today:
            body.append(f"   {t['symbol']:<10} PnL {self._sign(t.get('pnl',0))}  "
                        f"{t.get('r_multiple',0):+.2f}R")
        try:
            with open(path, "w") as f:
                f.write("\n".join(body) + "\n")
            logger.info("Daily report written to %s", path)
        except Exception:
            logger.exception("Failed to write daily report")
        return path

    # ------------------------------------------------------------------ #
    @staticmethod
    def _sign(x: float) -> str:
        return f"${x:+,.2f}"

    @staticmethod
    def _pct(x: Optional[float]) -> str:
        return "n/a" if x is None else f"{x*100:.0f}%"
