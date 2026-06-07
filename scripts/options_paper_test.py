"""Part 6 — find the best options trade available right now and preview it.

Runs the bot's real signal pipeline (technical + quant + regime -> master score)
over the optionable watchlist, picks the highest-conviction name, then builds the
ATM call/put plan the live agent *would* place when ``OPTIONS_ENABLED`` is on —
and prints exactly what it would look like, including the Telegram alert.

This is a dry run: it places no orders. Add ``--place`` to actually buy the
previewed contract on the paper account.

    python scripts/options_paper_test.py            # preview only
    python scripts/options_paper_test.py --place     # preview + buy on paper
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from config import settings
from src.data.feed import MarketFeed, is_crypto
from src.execution.broker import Broker
from src.signals.options_strategy import OptionsStrategy, OptionPosition, OptionPositionStore
from src.signals.quant import QuantAnalysis
from src.signals.regime import RegimeDetector
from src.signals.rr_filter import RRFilter
from src.signals.scorer import MasterScorer
from src.signals.strategy import Signal
from src.signals.technical import TechnicalAnalysis

BAR = "─" * 64


def main() -> None:
    place = "--place" in sys.argv
    k, s = settings.ALPACA_API_KEY, settings.ALPACA_SECRET_KEY
    if not k or not s:
        sys.exit("Missing ALPACA_API_KEY / ALPACA_SECRET_KEY in .env")

    feed = MarketFeed(k, s, stock_feed=settings.STOCK_DATA_FEED, cache_ttl=240)
    broker = Broker(k, s, paper=settings.PAPER)
    technical = TechnicalAnalysis()
    quant = QuantAnalysis()
    regime_detector = RegimeDetector()
    scorer = MasterScorer(min_score=settings.MIN_SCORE, rr_target=settings.RR_RATIO)
    rr_filter = RRFilter(rr_ratio=settings.RR_RATIO, atr_period=settings.ATR_PERIOD,
                         atr_multiplier=settings.ATR_MULTIPLIER,
                         swing_lookback=settings.SWING_LOOKBACK,
                         path_veto=settings.RR_PATH_VETO)
    options = OptionsStrategy(
        k, s, paper=settings.PAPER,
        dte_min=settings.OPTIONS_DTE_MIN, dte_max=settings.OPTIONS_DTE_MAX,
        risk_pct=settings.OPTIONS_RISK_PCT, profit_target=settings.OPTIONS_PROFIT_TARGET,
        stop_loss=settings.OPTIONS_STOP_LOSS, max_positions=settings.OPTIONS_MAX_POSITIONS,
        skip_earnings=settings.OPTIONS_SKIP_EARNINGS,
        expiry_exit_days=settings.OPTIONS_EXPIRY_EXIT_DAYS)

    equity = broker.get_equity()
    budget = equity * settings.OPTIONS_RISK_PCT
    print(BAR)
    print(f"OPTIONS PAPER TEST   equity ${equity:,.2f}   1% budget ${budget:,.0f}")
    print(f"gate score>={settings.OPTIONS_MIN_SCORE:.0f} · {settings.OPTIONS_DTE_MIN}-"
          f"{settings.OPTIONS_DTE_MAX} DTE · ATM · target +"
          f"{settings.OPTIONS_PROFIT_TARGET*100:.0f}% / stop -{settings.OPTIONS_STOP_LOSS*100:.0f}%")
    print(BAR)

    # Equities only (no crypto options on Alpaca).
    universe = [x for x in settings.load_watchlist() if not is_crypto(x)]
    bars = feed.get_bars_batch(universe, "1Day", settings.LOOKBACK_BARS)
    spy = feed.get_bars(settings.MARKET_PROXY, "1Day", settings.LOOKBACK_BARS)

    # Score every name with the deterministic core (technical+quant+regime+RR).
    ranked = []
    for sym in universe:
        df = bars.get(sym)
        if df is None or df.empty or len(df) < 60:
            continue
        tech = technical.analyze(df)
        if tech is None or tech.trend_bias == "neutral":
            continue
        side = tech.trend_bias
        q = quant.analyze(df, market_df=spy if not spy.empty else None)
        reg = regime_detector.detect(df, spy_df=spy if not spy.empty else None)
        price = float(df["close"].iloc[-1])
        sig = Signal(sym, side, price, tech.values.get("rsi14") or 50.0,
                     df.index[-1], tech.trend_bias)
        plan = rr_filter.evaluate(sig, df)
        sc = scorer.score(sym, side, technical=tech, quant=q, regime=reg, plan=plan)
        ranked.append((sc.total, sym, side, price))

    ranked.sort(reverse=True)
    if not ranked:
        sys.exit("No candidates produced a directional signal right now.")

    print("\nTop signals (master score):")
    for total, sym, side, price in ranked[:8]:
        flag = "✅ qualifies" if total >= settings.OPTIONS_MIN_SCORE else "·"
        arrow = "↑call" if side == "long" else "↓put"
        print(f"  {sym:6} {total:5.1f}  {arrow:6} ${price:>8.2f}  {flag}")

    # Best optionable trade = highest-scored name that yields a viable, affordable
    # plan. ranked is descending, so the first viable plan is the best one.
    chosen = None
    for total, sym, side, price in ranked:
        plan = options.plan_trade(sym, side, price, equity, total)
        if plan is not None:
            chosen = (total, sym, side, price, plan)
            break
    if chosen is None:
        sys.exit("\nNo viable/affordable ATM option found for any signal right now.")

    total, sym, side, price, plan = chosen
    q = plan.quote
    qualifies = total >= settings.OPTIONS_MIN_SCORE
    print("\n" + BAR)
    print(f"BEST OPTIONS TRADE RIGHT NOW   {'(LIVE-QUALIFYING)' if qualifies else '(BELOW 70 GATE — preview only)'}")
    print(BAR)
    print(f"  {plan.description}")
    print(f"  Signal           {sym} {side.upper()}  ·  master score {total:.1f}/100")
    print(f"  Contract         {q.symbol}")
    print(f"  Type / Strike    {q.type.upper()}  ${q.strike:,.2f}   (underlying ${price:,.2f})")
    print(f"  Expiration       {q.expiration}   ({q.dte} days out)")
    if q.delta is not None:
        print(f"  Greeks           delta {q.delta:+.2f}   IV {q.iv*100:.0f}%" if q.iv
              else f"  Greeks           delta {q.delta:+.2f}")
    print(f"  Premium          ${q.premium:.2f}/share  (bid ${q.bid:.2f} / ask ${q.ask:.2f})"
          f"  = ${q.contract_cost:,.0f}/contract")
    print(f"  Contracts        {plan.contracts}   (1% budget ${budget:,.0f})")
    print(f"  Total cost       ${plan.cost:,.0f}   = {plan.cost/equity*100:.2f}% of account")
    print(f"  Max loss         ${plan.risk_dollars:,.0f}   (the premium — defined risk)")
    print(f"  Take profit      premium ${plan.target_premium:.2f}  (+100%)  "
          f"-> ~+${plan.cost:,.0f} profit, position worth ~${plan.cost*2:,.0f}")
    print(f"  Stop             premium ${plan.stop_premium:.2f}  (-50%)   "
          f"-> ~-${plan.cost/2:,.0f}")

    print("\n  Telegram alert that would fire:")
    print("  ┌" + "─" * 56)
    from datetime import date
    d = date.fromisoformat(q.expiration)
    print(f"  │ *OPTION BOUGHT* 🎯")
    print(f"  │ {sym} {q.type.upper()} · Strike ${q.strike:,.0f} · "
          f"Exp {d.strftime('%b')} {d.day} · Paid ${plan.cost:,.0f}")
    print(f"  │ {plan.contracts} contract(s) @ ${q.premium:.2f}")
    print(f"  │ _{plan.description}_")
    print("  └" + "─" * 56)

    if not qualifies:
        print(f"\n  NOTE: master score {total:.1f} < {settings.OPTIONS_MIN_SCORE:.0f} gate — "
              f"the live bot would NOT place this. Shown so you can see the shape.")

    if place:
        if not qualifies:
            print("\nRefusing to --place a sub-gate trade. Re-run when a 70+ signal exists.")
            return
        print("\nPlacing the order on the paper account…")
        order = broker.buy_option(q.symbol, plan.contracts)
        if order is None:
            print("  order failed — see logs.")
            return
        store = OptionPositionStore(log_dir=settings.LOG_DIR)
        positions = store.load()
        positions[q.symbol] = OptionPosition(
            symbol=q.symbol, underlying=sym, type=q.type, strike=q.strike,
            expiration=q.expiration, contracts=plan.contracts, premium_paid=q.premium,
            cost_basis=plan.cost, side_bias=plan.side_bias, score=total,
            target_premium=plan.target_premium, stop_premium=plan.stop_premium,
            entry_time=datetime.now(timezone.utc).isoformat())
        store.save(positions)
        print(f"  ✓ bought {plan.contracts}x {q.symbol} (order {order.id})")
        print("  tracked in logs/option_positions.json — the live loop will manage it.")
    else:
        print("\n(Dry run — no order placed. Add --place to buy this on paper.)")


if __name__ == "__main__":
    main()
