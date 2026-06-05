"""End-of-day review — plain-English summary of the trading day.

Reads today's agent log (``logs/agent_YYYYMMDD.log``) and the live state snapshot
(``logs/agent_state.json``) and prints a colorful rich report:

* trades taken with outcome
* signals rejected and why
* the day's highest-scoring symbols
* current regime / market state
* win/loss record and running PnL
* one specific tuning recommendation derived from the day's patterns

Run:  python scripts/daily_review.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config import settings

console = Console()


def _log_path() -> str:
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    return os.path.join(settings.LOG_DIR, f"agent_{date}.log")


def _read_log() -> list[str]:
    path = _log_path()
    if not os.path.exists(path):
        return []
    with open(path, errors="ignore") as f:
        return f.readlines()


def _read_state() -> dict:
    path = os.path.join(settings.LOG_DIR, "agent_state.json")
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def parse_log(lines: list[str]) -> dict:
    entered, rejected, no_signal = [], [], 0
    reject_reasons = Counter()
    for ln in lines:
        if "ENTERED" in ln:
            m = re.search(r"(\S+): ENTERED score=([\d.]+) rr=([\d.]+)", ln)
            if m:
                entered.append((m.group(1), float(m.group(2)), float(m.group(3))))
        elif "blocked by risk" in ln:
            sym = ln.split(":")[3].strip().split()[0] if len(ln.split(":")) > 3 else "?"
            rejected.append(ln.strip())
            if "score" in ln:
                reject_reasons["low score"] += 1
            elif "RR" in ln:
                reject_reasons["RR < 5"] += 1
            elif "corr" in ln:
                reject_reasons["correlation"] += 1
            else:
                reject_reasons["other risk"] += 1
        elif "blocked by portfolio heat" in ln:
            reject_reasons["portfolio heat"] += 1
        elif "rejected, structural resistance" in ln:
            reject_reasons["path blocked (RR veto)"] += 1
        elif "no aligned signal" in ln or "no signal" in ln:
            no_signal += 1
    return {"entered": entered, "rejected": rejected,
            "reject_reasons": reject_reasons, "no_signal": no_signal}


def tuning_recommendation(parsed: dict, state: dict) -> str:
    rr = parsed["reject_reasons"]
    n_entered = len(parsed["entered"])
    closed = state.get("closed_today", [])
    wins = sum(1 for t in closed if t.get("pnl", 0) > 0)
    losses = len(closed) - wins

    if n_entered == 0 and rr.get("low score", 0) >= 3:
        return ("Most candidates scored just below the 70 gate and nothing traded. "
                "Consider lowering MIN_SCORE to ~65 for a few sessions and watch whether "
                "the extra trades are profitable — but only after validating on a backtest.")
    if rr.get("path blocked (RR veto)", 0) >= 3:
        return ("Many setups were vetoed because a swing level blocked the path to the 5:1 "
                "target. Consider widening SWING_LOOKBACK or testing RR_RATIO=4 to see if "
                "the looser target still clears the validation harness.")
    if losses > wins and len(closed) >= 3:
        return ("More losses than wins today. Before loosening anything, raise MIN_SCORE "
                "(e.g. 75) so only higher-conviction setups trade, and re-check the regime "
                "filter — losses in a 'ranging' regime suggest the momentum strategy is "
                "firing in the wrong environment.")
    if rr.get("correlation", 0) >= 2:
        return ("Several trades were blocked by correlation with open positions. That's the "
                "risk layer working — no change needed, but a more diversified watchlist "
                "would give the scanner more independent opportunities.")
    return ("Nothing stands out today. Keep the parameters fixed and let the sample grow — "
            "the validation harness needs ~100+ trades before any tuning is trustworthy.")


def main() -> None:
    lines = _read_log()
    state = _read_state()
    parsed = parse_log(lines)

    console.print(Panel.fit(
        f"[bold]DAILY REVIEW[/bold]  {datetime.now().strftime('%Y-%m-%d')}",
        style="cyan"))

    if not lines:
        console.print("[yellow]No agent log for today yet — has the agent run?[/yellow]")
    if not state:
        console.print("[yellow]No agent_state.json — start the agent to populate.[/yellow]")

    # --- Account / PnL ----------------------------------------------------- #
    eq = state.get("equity", 0.0)
    daily = state.get("daily_pnl", 0.0)
    weekly = state.get("weekly_pnl", 0.0)
    acct = Table.grid(padding=(0, 3))
    acct.add_row("Equity", f"[bold]${eq:,.2f}[/bold]")
    acct.add_row("Daily PnL", f"[{'green' if daily>=0 else 'red'}]${daily:+,.2f}[/]")
    acct.add_row("Weekly PnL", f"[{'green' if weekly>=0 else 'red'}]${weekly:+,.2f}[/]")
    acct.add_row("Market state", f"[cyan]{state.get('risk_state','unknown')}[/cyan]")
    console.print(Panel(acct, title="Account & PnL", border_style="green"))

    # --- Trades taken ------------------------------------------------------ #
    closed = state.get("closed_today", [])
    t = Table(title="Trades taken / closed today", title_style="bold")
    for c in ("Symbol", "PnL", "R", "Outcome"):
        t.add_column(c)
    for tr in closed:
        pnl = tr.get("pnl", 0)
        t.add_row(tr.get("symbol", "?"), f"${pnl:,.2f}", f"{tr.get('r_multiple',0):+.2f}",
                  "[green]WIN[/green]" if pnl > 0 else "[red]LOSS[/red]")
    if parsed["entered"]:
        for sym, sc, rr in parsed["entered"]:
            if sym not in {c.get("symbol") for c in closed}:
                t.add_row(sym, "[dim]open[/dim]", "—", f"[yellow]entered @{sc:.0f}[/yellow]")
    if not closed and not parsed["entered"]:
        t.add_row("—", "", "", "no trades")
    console.print(t)

    wins = sum(1 for c in closed if c.get("pnl", 0) > 0)
    losses = len(closed) - wins
    if closed:
        console.print(f"  Record: [green]{wins}W[/green] / [red]{losses}L[/red] "
                      f"({wins/len(closed)*100:.0f}% win rate)\n")

    # --- Rejections -------------------------------------------------------- #
    rj = Table(title="Why signals were rejected", title_style="bold")
    rj.add_column("Reason"); rj.add_column("Count", justify="right")
    for reason, n in parsed["reject_reasons"].most_common():
        rj.add_row(reason, str(n))
    if parsed["no_signal"]:
        rj.add_row("no aligned signal", str(parsed["no_signal"]))
    if not parsed["reject_reasons"] and not parsed["no_signal"]:
        rj.add_row("—", "0")
    console.print(rj)

    # --- Top scores -------------------------------------------------------- #
    scores = sorted(state.get("scores", []), key=lambda d: d.get("score", 0), reverse=True)[:5]
    if scores:
        ts = Table(title="Highest scores today", title_style="bold")
        ts.add_column("Symbol"); ts.add_column("Side"); ts.add_column("Score", justify="right")
        for s in scores:
            sc = s.get("score", 0)
            color = "green" if sc >= 70 else "yellow" if sc >= 50 else "red"
            ts.add_row(s["symbol"], s.get("side", ""), f"[{color}]{sc:.0f}[/]")
        console.print(ts)

    # --- Recommendation ---------------------------------------------------- #
    console.print(Panel(tuning_recommendation(parsed, state),
                        title="💡 Tuning recommendation", border_style="magenta"))


if __name__ == "__main__":
    main()
