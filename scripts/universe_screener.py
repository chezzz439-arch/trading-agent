"""Dynamic watchlist generator — screen the S&P 500 into a liquid universe.

Pulls S&P 500 constituents from Wikipedia, drops financials/banks (they trade on
their own rhythm), filters by price > $15, market cap > $3B and average volume >
1M shares, ranks by a dollar-liquidity score (avg volume x price), keeps the
TOP_N most liquid, appends crypto, and writes config/watchlist.json. Sends a
Telegram summary (count qualified + top 20, plus added/removed on change).

Run now:   python scripts/universe_screener.py
Schedule:  the agent re-runs this every Monday ~8am ET (see main.py); you can
           also cron it.
"""

from __future__ import annotations

import io
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
TOP_N = 50
CRYPTO = ["BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD", "MATIC/USD", "LINK/USD"]
MIN_PRICE = 15.0
MIN_MARKET_CAP = 3e9
MIN_AVG_VOLUME = 1e6
EXCLUDE_SECTORS = {"Financials"}          # banks/insurers behave differently
_OTC_MARKERS = ("OTC", "PNK", "PINK", "GREY")
WATCHLIST_PATH = settings.WATCHLIST_PATH


def sp500_table() -> pd.DataFrame:
    # Wikipedia 403s pandas' default urllib UA, so fetch with a browser UA.
    import requests
    html = requests.get(SP500_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
    return pd.read_html(io.StringIO(html))[0]


def non_financial_symbols() -> list[str]:
    table = sp500_table()
    sector_col = next((c for c in table.columns if "Sector" in str(c)), None)
    if sector_col is not None:
        table = table[~table[sector_col].isin(EXCLUDE_SECTORS)]
    return [str(s).strip() for s in table["Symbol"].tolist()]


def screen(symbols: list[str], top_n: int = TOP_N) -> list[dict]:
    """Return up to ``top_n`` qualifying rows, most liquid (volume x price) first."""
    import yfinance as yf

    yf_of = {s: s.replace(".", "-") for s in symbols}          # BRK.B -> BRK-B
    data = yf.download(list(yf_of.values()), period="3mo", interval="1d",
                       progress=False, auto_adjust=True)
    close, vol = data["Close"], data["Volume"]

    candidates = []
    for orig, yfs in yf_of.items():
        try:
            px = float(close[yfs].dropna().iloc[-1])
            av = float(vol[yfs].dropna().tail(20).mean())
        except Exception:
            continue
        if px > MIN_PRICE and av > MIN_AVG_VOLUME:
            candidates.append({"symbol": orig, "yf": yfs, "price": px,
                               "avg_volume": av, "liquidity": av * px})
    candidates.sort(key=lambda r: r["liquidity"], reverse=True)

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
    print(f"Screened {len(symbols)} non-financial S&P names -> {len(candidates)} pass "
          f"price+volume -> checked {checked} for mcap -> {len(qualified)} qualified.")
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
                     "min_avg_volume": MIN_AVG_VOLUME, "top_n": TOP_N,
                     "exclude_sectors": sorted(EXCLUDE_SECTORS),
                     "rank": "liquidity = avg_volume * price"},
        "count": len(symbols), "symbols": symbols,
    }
    tmp = WATCHLIST_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, WATCHLIST_PATH)


def main(show_top: int = 20) -> None:
    print("Pulling S&P 500 (excluding financials) from Wikipedia…")
    symbols = non_financial_symbols()
    qualified = screen(symbols, TOP_N)

    stock_syms = [r["symbol"] for r in qualified]
    watchlist = stock_syms + CRYPTO
    previous = load_existing()
    added = sorted(set(watchlist) - set(previous))
    removed = sorted(set(previous) - set(watchlist))
    save(watchlist)

    print(f"\n{len(stock_syms)}/{len(symbols)} non-financial S&P stocks qualified.")
    print(f"Saved {len(watchlist)} symbols ({len(stock_syms)} stocks + {len(CRYPTO)} crypto) "
          f"to {WATCHLIST_PATH}.")
    print(f"\nTop {show_top} by liquidity (avg volume x price):")
    print(f"  {'#':>3} {'SYM':<8}{'price':>10}{'mkt cap':>10}{'$ liquidity/day':>18}")
    for i, r in enumerate(qualified[:show_top], 1):
        mc = f"${r['market_cap']/1e9:.0f}B" if r.get("market_cap") else "n/a"
        print(f"  {i:>3} {r['symbol']:<8}{r['price']:>10.2f}{mc:>10}{r['liquidity']/1e9:>15.1f}B")

    try:
        from src.monitoring.telegram_bot import TelegramNotifier
        n = TelegramNotifier()
        if n.enabled:
            top20 = ", ".join(r["symbol"] for r in qualified[:20])
            msg = (f"*WATCHLIST REFRESHED* 🔄\n{len(stock_syms)} of {len(symbols)} "
                   f"non-financial S&P stocks qualified (+{len(CRYPTO)} crypto).\n"
                   f"*Top 20:* {top20}")
            if previous:
                msg += (f"\nAdded ({len(added)}): {', '.join(added[:10]) or 'none'}"
                        f"\nRemoved ({len(removed)}): {', '.join(removed[:10]) or 'none'}")
            n.send(msg)
            print("\nTelegram summary sent.")
    except Exception:
        pass


if __name__ == "__main__":
    main()
