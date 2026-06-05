"""Layer 3 — Compact terminal dashboard (rich).

Reads the agent's shared state (``StateStore``) and renders a live, colored,
single-screen view that refreshes every few seconds. Designed to sit in a
terminal next to the running agent.

Run standalone:  ``python -m src.monitoring.terminal_dash``
"""

from __future__ import annotations

import time
from typing import Optional

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.monitoring.state_store import StateStore


def _money(x: float) -> Text:
    return Text(f"${x:,.2f}", style="green" if x >= 0 else "red")


def _score_bar(score: float, width: int = 20) -> Text:
    filled = int(round(score / 100 * width))
    color = "green" if score >= 70 else "yellow" if score >= 50 else "red"
    bar = "█" * filled + "░" * (width - filled)
    return Text(bar, style=color)


class TerminalDashboard:
    def __init__(self, state_store: Optional[StateStore] = None, refresh: int = 5):
        self.store = state_store or StateStore()
        self.refresh = refresh
        self.console = Console()

    # ------------------------------------------------------------------ #
    def render(self, state: dict):
        if not state:
            return Panel(Text("Waiting for agent state… (is the agent running?)",
                              style="dim"), title="Trading Agent")

        equity = state.get("equity", 0.0)
        bp = state.get("buying_power", 0.0)
        daily = state.get("daily_pnl", 0.0)
        halted = state.get("halted", False)
        regime = state.get("risk_state", "unknown")

        header = Table.grid(expand=True)
        header.add_column(justify="left")
        header.add_column(justify="right")
        status = Text("● HALTED", style="bold red") if halted else Text("● ACTIVE", style="bold green")
        header.add_row(Text(f"Equity ${equity:,.2f}", style="bold"),
                       Text(f"Buying Power ${bp:,.2f}"))
        header.add_row(Text("Daily PnL ").append(_money(daily)),
                       Text("Market: ").append(Text(regime, style="cyan")).append("  ").append(status))

        # Positions
        pos_tbl = Table(title="Open Positions", expand=True, title_style="bold")
        for col in ("Symbol", "Side", "Qty", "Unreal PnL", "%"):
            pos_tbl.add_column(col)
        for p in state.get("open_positions", []):
            pnl = p.get("pnl", 0.0)
            pos_tbl.add_row(str(p.get("symbol")), str(p.get("side", "")), str(p.get("qty", "")),
                            _money(pnl), Text(f"{p.get('pnl_pct',0):+.2f}%",
                            style="green" if pnl >= 0 else "red"))
        if not state.get("open_positions"):
            pos_tbl.add_row("—", "", "", "", "")

        # Scores mini bar chart
        score_tbl = Table(title="Signal Scores", expand=True, title_style="bold")
        score_tbl.add_column("Symbol", width=10)
        score_tbl.add_column("Score", justify="right", width=6)
        score_tbl.add_column("", ratio=1)
        for s in sorted(state.get("scores", []), key=lambda d: d.get("score", 0), reverse=True):
            sc = s.get("score", 0)
            flag = " ◀ TRADE" if s.get("passed") else ""
            score_tbl.add_row(f"{s.get('symbol')} {s.get('side','')[:1].upper()}",
                              Text(f"{sc:.0f}", style="bold"),
                              Text.assemble(_score_bar(sc), Text(flag, style="bold green")))
        if not state.get("scores"):
            score_tbl.add_row("—", "", "")

        return Group(
            Panel(header, title="Trading Agent", border_style="red" if halted else "green"),
            pos_tbl,
            score_tbl,
            Text(f"updated {state.get('updated_at','?')}", style="dim"),
        )

    def run(self) -> None:
        with Live(self.render(self.store.read_state()), console=self.console,
                  refresh_per_second=4, screen=False) as live:
            try:
                while True:
                    time.sleep(self.refresh)
                    live.update(self.render(self.store.read_state()))
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    TerminalDashboard().run()
