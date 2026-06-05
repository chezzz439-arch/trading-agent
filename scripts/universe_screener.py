"""Dynamic watchlist generator — screen the S&P 500 into a liquid universe.

Pulls the S&P 500 constituents from Wikipedia, filters by price > $15,
market cap > $3B and average volume > 1M shares (dropping OTC/pink names),
keeps the ``TOP_N`` most liquid, appends the crypto symbols, and writes the
result to ``config/watchlist.json``. Sends a Telegram message when the list
changes.

Run now:        python scripts/universe_screener.py
Schedule:       Monday ~7:00am ET (cron or the schedule skill) to refresh weekly.

Efficiency: price/volume come from one batched download; market cap is fetched
via yfinance fast_info only for candidates that already pass price+volume, in
volume order, until TOP_N qualify.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from config import settings

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
TOP_N = 200
CRYPTO = ["BTC/USD", "ETH/USD", "SOL/USD"]
MIN_PRICE = 15.0
MIN_MARKET_CAP = 3e9
MIN_AVG_VOLUME = 1e6
_OTC_MARKERS = ("OTC", "PNK", "PINK", "GREY")
WATCHLIST_PATH = os.path.join("config", "watchlist.json")


def sp500_symbols() -> list[str]:
    # Wikipedia 403s pandas' default urllib UA, so fetch with a browser UA.
    import io
    import requests
    html = requests.get(SP500_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
    table = pd.read_html(io.StringIO(html))[0]
    return [str(s).strip() for s in table["Symbol"].tolist()]


def screen(symbols: list[str], top_n: int = TOP_N) -> list[dict]:
    """Return up to ``top_n`` qualifying rows, most liquid first."""
    import yfinance as yf

    # Wikipedia uses dotted class tickers (BRK.B); yfinance wants dashes.
    yf_of = {s: s.replace(".", "-") for s in symbols}
    data = yf.download(list(yf_of.values()), period="3mo", interval="1d",
                       progress=False, auto_adjust=True)
    close, vol = data["Close"], data["Volume"]

    # Pass 1 (free): price + volume from the batch download.
    candidates = []
    for orig, yfs in yf_of.items():
        try:
            px = float(close[yfs].dropna().iloc[-1])
            av = float(vol[yfs].dropna().tail(20).mean())
        except Exception:
            continue
        if px > MIN_PRICE and av > MIN_AVG_VOLUME:
            candidates.append({"symbol": orig, "yf": yfs, "price": px, "avg_volume": av})
    candidates.sort(key=lambda r: r["avg_volume"], reverse=True)

    # Pass 2: market-cap + exchange via fast_info, in volume order, until top_n.
    qualified, checked = [], 0
    for r in candidates:
        if len(qualified) >= top_n:
            break
        checked += 1
        try:
            fi = yf.Ticker(r["yf"]).fast_info
            mcap = getattr(fi, "market_cap", None)
            exch = (getattr(fi, "exchange", "") or "").upper()
        except Exception:
            mcap, exch = None, ""
        if any(m in exch for m in _OTC_MARKERS):
            continue
        if mcap is not None and mcap < MIN_MARKET_CAP:
            continue
        r["market_cap"] = mcap
        r["exchange"] = exch
        qualified.append(r)
    print(f"Screened {len(symbols)} S&P names -> {len(candidates)} pass price+volume "
          f"-> checked {checked} for mcap -> {len(qualified)} qualified (top {top_n}).")
    return qualified


def load_existing() -> list[str]:
    try:
        with open(WATCHLIST_PATH) as f:
            return json.load(f).get("symbols", [])
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save(symbols: list[str]) -> None:
    os.makedirs(os.path.dirname(WATCHLIST_PATH), exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "criteria": {"min_price": MIN_PRICE, "min_market_cap": MIN_MARKET_CAP,
                     "min_avg_volume": MIN_AVG_VOLUME, "top_n": TOP_N},
        "count": len(symbols),
        "symbols": symbols,
    }
    tmp = WATCHLIST_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, WATCHLIST_PATH)


def main(show_top: int = 20) -> None:
    print("Pulling S&P 500 constituents from Wikipedia…")
    symbols = sp500_symbols()
    qualified = screen(symbols, TOP_N)

    stock_syms = [r["symbol"] for r in qualified]
    watchlist = stock_syms + CRYPTO

    previous = load_existing()
    changed = set(watchlist) != set(previous)
    save(watchlist)
    print(f"\nSaved {len(watchlist)} symbols to {WATCHLIST_PATH} "
          f"({len(stock_syms)} stocks + {len(CRYPTO)} crypto). "
          f"{'CHANGED' if changed else 'unchanged'} vs previous.")

    print(f"\nTop {show_top} most-liquid qualifying stocks:")
    print(f"  {'#':>3} {'SYM':<8}{'price':>10}{'avg vol':>15}{'mkt cap':>12}{'exch':>6}")
    for i, r in enumerate(qualified[:show_top], 1):
        mc = f"${r['market_cap']/1e9:.0f}B" if r.get("market_cap") else "n/a"
        print(f"  {i:>3} {r['symbol']:<8}{r['price']:>10.2f}{r['avg_volume']:>15,.0f}"
              f"{mc:>12}{r.get('exchange',''):>6}")

    # Telegram notice on change.
    try:
        from src.monitoring.telegram_bot import TelegramNotifier
        n = TelegramNotifier()
        if n.enabled and changed:
            added = sorted(set(watchlist) - set(previous))[:10]
            removed = sorted(set(previous) - set(watchlist))[:10]
            n.send(f"*WATCHLIST UPDATED* 🔄\n{len(watchlist)} symbols "
                   f"({len(stock_syms)} stocks + crypto)\n"
                   f"Added: {', '.join(added) or 'none'}\n"
                   f"Removed: {', '.join(removed) or 'none'}")
            print("\nTelegram update sent.")
    except Exception:
        pass


if __name__ == "__main__":
    main()
