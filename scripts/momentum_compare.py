"""Old-vs-new backtest comparison for the pure-momentum-long-only refactor.

Runs the full scorer-gated pipeline (long-only, score>=70, hybrid target,
realistic costs) across the whole watchlist over 4y, collects every trade's
R-multiple, and reports expectancy / win-rate / total-R plus a bootstrap
p-value on expectancy (identical method for both runs, so the comparison is
apples-to-apples).

  python scripts/momentum_compare.py old     # run BEFORE the refactor
  python scripts/momentum_compare.py new     # run AFTER  the refactor
  python scripts/momentum_compare.py compare # print old-vs-new table

Writes logs/momentum_<label>.json. Parallel + file output (run_pipeline does
quant per-bar, so never run this serially or pipe it to tail).
"""

from __future__ import annotations

import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

import numpy as np

from config import settings
from src.backtest.costs import CostModel
from src.backtest.engine import Backtester
from src.signals.rr_filter import RRFilter

PERIOD = "4y"
MIN_SCORE = 70.0
N_BOOT = 10000


def _collect(args):
    """Long-only pipeline backtest for one symbol -> list of long R-multiples."""
    symbol, df, spy = args
    try:
        rrf = RRFilter(rr_ratio=settings.RR_RATIO, atr_period=settings.ATR_PERIOD,
                       atr_multiplier=settings.ATR_MULTIPLIER,
                       swing_lookback=settings.SWING_LOOKBACK,
                       path_veto=settings.RR_PATH_VETO, hybrid=True)
        bt = Backtester(rr_filter=rrf, cost_model=CostModel.equities())
        bt.load_data = lambda s, period="2y", interval="1d": (spy if s == "SPY" else df)
        res = bt.run_pipeline(symbol, period=PERIOD, min_score=MIN_SCORE)
        if res is None or res.trades is None or res.trades.empty:
            return []
        t = res.trades
        # long-only: the new scorer vetoes shorts; filter old the same way.
        return [float(r) for sd, r in zip(t["side"], t["r_multiple"]) if str(sd) == "long"]
    except Exception:
        return []


def _bootstrap_p(rs: list[float]) -> float:
    """P(mean R <= 0) under bootstrap resampling — small = real positive edge."""
    if len(rs) < 5:
        return 1.0
    arr = np.array(rs)
    n = len(arr)
    # deterministic generator (Date/Random are fine in a plain script, not a workflow)
    rng = np.random.default_rng(12345)
    means = arr[rng.integers(0, n, size=(N_BOOT, n))].mean(axis=1)
    return float(np.mean(means <= 0.0))


def run(label: str):
    syms = settings.load_watchlist()
    loader = Backtester()
    cache = {}
    for s in syms + ["SPY"]:
        d = loader.load_data(s, period=PERIOD, interval="1d")
        if d is not None and not d.empty:
            cache[s] = d
    spy = cache["SPY"]
    stocks = [s for s in syms if s in cache and s != "SPY" and "/" not in s]
    print(f"[{label}] collecting trades across {len(stocks)} stocks ({PERIOD})…")
    with Pool(min(os.cpu_count() or 2, 10)) as pool:
        per = pool.map(_collect, [(s, cache[s], spy) for s in stocks])

    rs = [r for lst in per for r in lst]
    n_sym_profitable = sum(1 for lst in per if lst and float(np.mean(lst)) > 0)
    n_sym_traded = sum(1 for lst in per if lst)
    out = {
        "label": label, "period": PERIOD, "min_score": MIN_SCORE,
        "symbols_traded": n_sym_traded,
        "symbols_profitable": n_sym_profitable,
        "trades": len(rs),
        "win_rate": round(float(np.mean([r > 0 for r in rs])) * 100, 1) if rs else 0.0,
        "expectancy_R": round(float(np.mean(rs)), 4) if rs else 0.0,
        "total_R": round(float(np.sum(rs)), 1) if rs else 0.0,
        "avg_win_R": round(float(np.mean([r for r in rs if r > 0])), 2) if any(r > 0 for r in rs) else 0.0,
        "avg_loss_R": round(float(np.mean([r for r in rs if r <= 0])), 2) if any(r <= 0 for r in rs) else 0.0,
        "bootstrap_p": round(_bootstrap_p(rs), 4),
    }
    path = os.path.join("logs", f"momentum_{label}.json")
    os.makedirs("logs", exist_ok=True)
    json.dump(out, open(path, "w"), indent=2)
    print(json.dumps(out, indent=2))
    print(f"[saved -> {path}]")


def compare():
    o = json.load(open("logs/momentum_old.json"))
    n = json.load(open("logs/momentum_new.json"))
    rows = [("trades", "trades"), ("symbols_traded", "symbols_traded"),
            ("symbols_profitable", "symbols_profitable"), ("win_rate", "win% "),
            ("expectancy_R", "expectancy R/trade"), ("total_R", "total R"),
            ("avg_win_R", "avg win R"), ("avg_loss_R", "avg loss R"),
            ("bootstrap_p", "bootstrap p (edge)")]
    print(f"\n{'metric':22} {'OLD complex':>14} {'NEW momentum':>14}")
    print("-" * 52)
    for key, lab in rows:
        print(f"{lab:22} {str(o[key]):>14} {str(n[key]):>14}")
    print("-" * 52)
    de = n["expectancy_R"] - o["expectancy_R"]
    print(f"\nExpectancy change: {de:+.4f} R/trade "
          f"({'IMPROVED' if de > 0 else 'WORSE'})")
    print(f"Edge significance: old p={o['bootstrap_p']} -> new p={n['bootstrap_p']} "
          f"({'IMPROVED' if n['bootstrap_p'] < o['bootstrap_p'] else 'WORSE/SAME'})")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "old"
    if arg == "compare":
        compare()
    else:
        run(arg)
