"""Dynamic watchlist generator — screen a combined US equity universe.

Combines S&P 500 + S&P 400 (mid-cap) + Nasdaq-100 + Russell 1000 from Wikipedia,
de-duplicates, tags each with its sector, filters by price > $15, market cap >
$3B, avg volume > 500k and >= 1 year of history, ranks by a liquidity score
(price x volume x market cap), and keeps the top 100. Appends 20 crypto pairs,
tagging each tradable/data-only based on a live Alpaca check, with category and
(for meme coins) a smaller size override.

Writes config/watchlist.json with a flat ``symbols`` list (backward compatible)
plus a ``meta`` map (name/sector/asset_class/tradable/category/...). Sends a
Telegram summary. Run now or weekly (the agent re-runs it Monday ~8am ET).
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

MIN_PRICE = 15.0
MIN_MARKET_CAP = 3e9
MIN_AVG_VOLUME = 500_000
MIN_HISTORY_DAYS = 240          # ~1 trading year
TOP_N_STOCKS = 150
WATCHLIST_PATH = settings.WATCHLIST_PATH

_SOURCES = [
    ("S&P 500", "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", 0, "Symbol", "Security", "GICS Sector"),
    ("S&P 400", "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", 0, "Symbol", "Security", "GICS Sector"),
    ("Nasdaq-100", "https://en.wikipedia.org/wiki/Nasdaq-100", 5, "Ticker", "Company", None),
    ("Russell 1000", "https://en.wikipedia.org/wiki/Russell_1000_Index", 3, "Symbol", "Company", "GICS Sector"),
]

_SECTOR_FIX = {"Information Technology": "Technology", "Health Care": "Healthcare"}

# 23 crypto pairs with category; DOGE gets a smaller size override.
# (Tradability is checked live against Alpaca; untradeable pairs are tagged
# data-only automatically. MKR/SNX were requested but are inactive/unlisted on
# Alpaca, so they're omitted rather than added as never-tradeable noise.)
_CRYPTO = [
    ("BTC/USD", "Bitcoin", "Layer1"), ("ETH/USD", "Ethereum", "Layer1"),
    ("SOL/USD", "Solana", "Layer1"), ("BNB/USD", "Binance Coin", "Exchange"),
    ("XRP/USD", "Ripple", "Payments"), ("ADA/USD", "Cardano", "Layer1"),
    ("AVAX/USD", "Avalanche", "Layer1"), ("DOGE/USD", "Dogecoin", "Meme"),
    ("MATIC/USD", "Polygon", "Infrastructure"), ("DOT/USD", "Polkadot", "Infrastructure"),
    ("LINK/USD", "Chainlink", "DeFi"), ("UNI/USD", "Uniswap", "DeFi"),
    ("LTC/USD", "Litecoin", "Payments"), ("ATOM/USD", "Cosmos", "Infrastructure"),
    ("FIL/USD", "Filecoin", "Infrastructure"), ("NEAR/USD", "NEAR Protocol", "Layer1"),
    ("ARB/USD", "Arbitrum", "Infrastructure"), ("OP/USD", "Optimism", "Infrastructure"),
    ("APT/USD", "Aptos", "Layer1"), ("INJ/USD", "Injective", "DeFi"),
    ("AAVE/USD", "Aave", "DeFi"), ("CRV/USD", "Curve", "DeFi"),
    ("SUSHI/USD", "SushiSwap", "DeFi"),
]


def _fetch_tables(url):
    import requests
    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
    return pd.read_html(io.StringIO(html))


def combined_universe() -> dict[str, dict]:
    """Return {symbol: {name, sector}} merged across all index sources."""
    uni: dict[str, dict] = {}
    for name, url, idx, sym_col, name_col, sec_col in _SOURCES:
        try:
            t = _fetch_tables(url)[idx]
            cols = {str(c): c for c in t.columns}
            sc = cols.get(sym_col)
            nc = cols.get(name_col)
            secc = cols.get(sec_col) if sec_col else None
            added = 0
            for _, row in t.iterrows():
                s = str(row[sc]).strip().upper()
                if not s or s == "NAN":
                    continue
                if s not in uni:
                    sector = str(row[secc]).strip() if secc is not None else "Unknown"
                    sector = _SECTOR_FIX.get(sector, sector)
                    uni[s] = {"name": str(row[nc]).strip() if nc else s, "sector": sector}
                    added += 1
                elif uni[s]["sector"] in ("Unknown", "nan") and secc is not None:
                    uni[s]["sector"] = _SECTOR_FIX.get(str(row[secc]).strip(), str(row[secc]).strip())
            print(f"  {name}: {len(t)} rows, +{added} new (universe now {len(uni)})")
        except Exception as e:
            print(f"  {name}: FAILED ({e}) — skipping")
    return uni


def screen_stocks(uni: dict[str, dict]) -> tuple[list[dict], dict]:
    """Two-pass screen. Returns (top_qualified_rows, funnel_counts)."""
    import yfinance as yf

    symbols = list(uni.keys())
    yf_of = {s: s.replace(".", "-") for s in symbols}
    funnel = {"universe": len(symbols), "have_data": 0, "price_ok": 0,
              "volume_ok": 0, "history_ok": 0, "mcap_ok": 0}

    # Pass 1: chunked batch download of 1y daily bars -> price/volume/history.
    candidates = []
    chunk = 150
    yfs = list(yf_of.values())
    rev = {v: k for k, v in yf_of.items()}
    for i in range(0, len(yfs), chunk):
        part = yfs[i:i + chunk]
        try:
            data = yf.download(part, period="1y", interval="1d", progress=False,
                               auto_adjust=True, threads=True)
            close, vol = data["Close"], data["Volume"]
        except Exception:
            continue
        for yfs_sym in part:
            try:
                c = close[yfs_sym].dropna() if hasattr(close, "columns") else close.dropna()
                v = vol[yfs_sym].dropna() if hasattr(vol, "columns") else vol.dropna()
            except Exception:
                continue
            if len(c) < 5:
                continue
            funnel["have_data"] += 1
            px = float(c.iloc[-1]); av = float(v.tail(20).mean()); hist = len(c)
            if px <= MIN_PRICE:
                continue
            funnel["price_ok"] += 1
            if av <= MIN_AVG_VOLUME:
                continue
            funnel["volume_ok"] += 1
            if hist < MIN_HISTORY_DAYS:
                continue
            funnel["history_ok"] += 1
            orig = rev[yfs_sym]
            candidates.append({"symbol": orig, "yf": yfs_sym, "price": px,
                               "avg_volume": av, "history": hist,
                               "name": uni[orig]["name"], "sector": uni[orig]["sector"]})
        print(f"  …screened {min(i+chunk, len(yfs))}/{len(yfs)} "
              f"(passed price+vol+history so far: {len(candidates)})")

    # Pass 2: market cap (fast_info) on the most-liquid candidates first.
    candidates.sort(key=lambda r: r["avg_volume"] * r["price"], reverse=True)
    qualified = []
    for r in candidates:
        if len(qualified) >= TOP_N_STOCKS:
            break
        try:
            mcap = getattr(yf.Ticker(r["yf"]).fast_info, "market_cap", None)
        except Exception:
            mcap = None
        if mcap is None or mcap < MIN_MARKET_CAP:
            continue
        funnel["mcap_ok"] += 1
        r["market_cap"] = mcap
        r["liquidity"] = r["price"] * r["avg_volume"] * mcap
        qualified.append(r)
    qualified.sort(key=lambda r: r["liquidity"], reverse=True)
    return qualified[:TOP_N_STOCKS], funnel


def alpaca_tradable_crypto() -> set[str]:
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import GetAssetsRequest
        from alpaca.trading.enums import AssetClass
        tc = TradingClient(settings.ALPACA_API_KEY, settings.ALPACA_SECRET_KEY, paper=True)
        assets = tc.get_all_assets(GetAssetsRequest(asset_class=AssetClass.CRYPTO))
        return {a.symbol for a in assets if a.tradable}
    except Exception:
        return set()


def build_meta(stocks: list[dict], tradable: set[str]) -> dict:
    meta = {}
    for r in stocks:
        meta[r["symbol"]] = {"name": r["name"], "sector": r["sector"],
                             "asset_class": "stock", "tradable": True,
                             "price": round(r["price"], 2),
                             "market_cap": r.get("market_cap"),
                             "avg_volume": int(r["avg_volume"])}
    for sym, name, cat in _CRYPTO:
        meta[sym] = {"name": name, "sector": "Crypto", "asset_class": "crypto",
                     "category": cat, "tradable": sym in tradable}
        if cat == "Meme":
            meta[sym]["size_override"] = 0.005
    return meta


def save(symbols, meta, funnel):
    os.makedirs(os.path.dirname(WATCHLIST_PATH), exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "criteria": {"min_price": MIN_PRICE, "min_market_cap": MIN_MARKET_CAP,
                     "min_avg_volume": MIN_AVG_VOLUME, "min_history_days": MIN_HISTORY_DAYS,
                     "top_n_stocks": TOP_N_STOCKS, "rank": "price * volume * market_cap"},
        "funnel": funnel, "count": len(symbols), "symbols": symbols, "meta": meta,
    }
    tmp = WATCHLIST_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, WATCHLIST_PATH)


def main(show_top: int = 100) -> None:
    import time
    t0 = time.time()
    print("Building combined universe (S&P 500 + S&P 400 + Nasdaq-100 + Russell 1000)…")
    uni = combined_universe()
    print(f"\nScreening {len(uni)} unique stocks…")
    stocks, funnel = screen_stocks(uni)

    tradable = alpaca_tradable_crypto()
    crypto_syms = [c[0] for c in _CRYPTO]
    symbols = [r["symbol"] for r in stocks] + crypto_syms
    meta = build_meta(stocks, tradable)

    prev = []
    try:
        prev = json.load(open(WATCHLIST_PATH)).get("symbols", [])
    except Exception:
        pass
    added = sorted(set(symbols) - set(prev)); removed = sorted(set(prev) - set(symbols))
    save(symbols, meta, funnel)

    print("\n===== SCREEN FUNNEL =====")
    print(f"  Combined universe (deduped) : {funnel['universe']}")
    print(f"  Had usable price data       : {funnel['have_data']}")
    print(f"  Price > ${MIN_PRICE:.0f}             : {funnel['price_ok']}")
    print(f"  Volume > {MIN_AVG_VOLUME:,.0f}        : {funnel['volume_ok']}")
    print(f"  >= 1yr history              : {funnel['history_ok']}")
    print(f"  Market cap > $3B (top-checked): {funnel['mcap_ok']}")
    print(f"  FINAL stocks kept           : {len(stocks)}")

    n_trad = sum(1 for c in crypto_syms if c in tradable)
    print(f"\nCrypto: {n_trad}/{len(crypto_syms)} Alpaca-tradable, "
          f"{len(crypto_syms)-n_trad} data-only.")
    print(f"Total watchlist: {len(symbols)} ({len(stocks)} stocks + {len(crypto_syms)} crypto)")

    print(f"\nTop {show_top} stocks by liquidity (price x volume x market cap):")
    print(f"  {'#':>3} {'TICKER':<7}{'COMPANY':<26}{'SECTOR':<22}{'PRICE':>9}{'MKT CAP':>10}{'AVG VOL':>14}")
    for i, r in enumerate(stocks[:show_top], 1):
        mc = f"${r['market_cap']/1e9:.0f}B"
        print(f"  {i:>3} {r['symbol']:<7}{r['name'][:24]:<26}{r['sector'][:20]:<22}"
              f"{r['price']:>9.2f}{mc:>10}{r['avg_volume']:>14,.0f}")

    dt = time.time() - t0
    print(f"\nScreener completed in {dt:.0f}s.")

    try:
        from src.monitoring.telegram_bot import TelegramNotifier
        n = TelegramNotifier()
        if n.enabled:
            top10 = ", ".join(r["symbol"] for r in stocks[:10])
            msg = (f"*WATCHLIST EXPANDED* 🌐\n{len(stocks)} stocks (from {funnel['universe']} "
                   f"screened) + {len(crypto_syms)} crypto ({n_trad} tradable, "
                   f"{len(crypto_syms)-n_trad} data-only) = {len(symbols)} symbols.\n"
                   f"*Top 10:* {top10}")
            if prev:
                msg += f"\nAdded {len(added)}, removed {len(removed)} vs last list."
            n.send(msg)
            print("Telegram summary sent.")
    except Exception:
        pass


if __name__ == "__main__":
    main()
