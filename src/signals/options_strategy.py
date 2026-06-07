"""Options strategy — turn a strong stock signal into a long call/put.

When the master scorer produces a high-conviction signal the agent can express
it with a defined-risk **long option** instead of the stock:

* strong **LONG**  (score >= gate)  ->  buy an **ATM call**
* strong **SHORT** (score >= gate)  ->  buy an **ATM put**

Design (mirrors ``position_manager``): the **pure decision logic** — ATM
selection, premium sizing, and the take-profit / stop / expiry evaluation — is
separated from the **network layer** (Alpaca contract listing + option market
data, and the yfinance earnings calendar) so the core is fully unit-testable
offline and behaves identically in a paper test and the live loop.

Rules encoded here (all configurable in ``settings``):

* 30–45 days to expiration.
* At-the-money: strike closest to the current underlying price.
* Risk = 1% of equity, spent on **premium** (cost = price x 100 x contracts).
* Take profit at +100% (premium doubles); stop at -50% of premium paid.
* Never hold through an earnings date (skip if earnings falls before expiry).
* Max 3 concurrent option positions.

Long options only — no shorting/writing — so max loss is always the premium
paid, which is what the 1% sizing budgets for.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# Friendly names for the plain-English bet line ("Betting Apple goes UP …").
# Falls back to the ticker for anything not listed.
_NAMES = {
    "AAPL": "Apple", "TSLA": "Tesla", "NVDA": "Nvidia", "MSFT": "Microsoft",
    "AMZN": "Amazon", "META": "Meta", "GOOGL": "Google", "AMD": "AMD",
    "SPY": "the S&P 500", "QQQ": "the Nasdaq", "NFLX": "Netflix",
    "HOOD": "Robinhood", "UBER": "Uber", "BSX": "Boston Scientific",
    "CSCO": "Cisco", "PANW": "Palo Alto Networks",
}


def friendly_name(symbol: str) -> str:
    return _NAMES.get(symbol.upper(), symbol.upper())


# ============================================================================ #
# Lightweight value objects (no SDK types leak past this module)
# ============================================================================ #
@dataclass
class OptionQuote:
    """A selectable contract plus its live premium (per-share, x100 = per-contract)."""
    symbol: str            # OCC symbol, e.g. AAPL260710C00200000
    underlying: str
    type: str              # "call" | "put"
    strike: float
    expiration: str        # ISO date
    premium: float         # per-share mid (or last) price; contract cost = premium*100
    bid: float = 0.0
    ask: float = 0.0
    delta: Optional[float] = None
    iv: Optional[float] = None

    @property
    def dte(self) -> int:
        return (date.fromisoformat(self.expiration) - date.today()).days

    @property
    def contract_cost(self) -> float:
        return self.premium * 100.0


@dataclass
class OptionTradePlan:
    """A fully-sized, ready-to-place option order with its exit levels."""
    quote: OptionQuote
    contracts: int
    side_bias: str          # "up" (call) | "down" (put)
    score: float
    cost: float             # total premium outlay = premium*100*contracts
    target_premium: float   # sell here for the +100% profit target
    stop_premium: float     # cut here for the -50% stop
    risk_dollars: float     # max loss = the premium paid (defined risk)

    @property
    def description(self) -> str:
        return describe_bet(self.quote, self.side_bias)


@dataclass
class OptionPosition:
    """Persisted state for one open option position (current value comes live)."""
    symbol: str             # OCC symbol
    underlying: str
    type: str               # call | put
    strike: float
    expiration: str         # ISO date
    contracts: int
    premium_paid: float     # per-share entry premium
    cost_basis: float       # premium_paid * 100 * contracts
    side_bias: str          # up | down
    score: float = 0.0
    target_premium: float = 0.0
    stop_premium: float = 0.0
    entry_time: str = ""
    # alert/lifecycle flags so we notify each milestone exactly once
    up50_alerted: bool = False
    status: str = "open"    # open | closed

    @property
    def description(self) -> str:
        return describe_bet_fields(self.underlying, self.type, self.expiration)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "OptionPosition":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ============================================================================ #
# Pure decision logic (no network) — unit-testable
# ============================================================================ #
def select_atm(quotes: list[OptionQuote], underlying_price: float) -> Optional[OptionQuote]:
    """Pick the at-the-money contract: strike closest to the underlying price.

    Ties (equidistant strikes) break toward the lower strike for calls' favour;
    we simply take the minimum absolute distance, then the lower strike.
    """
    priced = [q for q in quotes if q.premium and q.premium > 0]
    if not priced:
        return None
    return min(priced, key=lambda q: (abs(q.strike - underlying_price), q.strike))


def size_contracts(premium: float, equity: float, risk_pct: float) -> int:
    """How many whole contracts fit in the premium budget (1% of equity).

    Each contract costs ``premium * 100``. Returns 0 when even one contract
    exceeds the budget (caller skips the trade).
    """
    if premium <= 0 or equity <= 0:
        return 0
    budget = equity * risk_pct
    per_contract = premium * 100.0
    return int(budget // per_contract)


def exit_decision(
    pos: OptionPosition,
    current_premium: float,
    *,
    profit_target: float,
    stop_loss: float,
    today: Optional[date] = None,
    expiry_exit_days: int = 1,
) -> tuple[str, str]:
    """Decide what to do with an open option given its current premium.

    Returns ``(action, reason)`` where action is one of:
      * ``"take_profit"`` — premium gained >= profit_target (e.g. doubled)
      * ``"stop"``        — premium lost >= stop_loss (e.g. -50%)
      * ``"expiry"``      — within ``expiry_exit_days`` of expiration
      * ``"hold"``        — none of the above
    """
    today = today or date.today()
    paid = pos.premium_paid
    if paid <= 0:
        return "hold", "no cost basis"

    gain = (current_premium - paid) / paid          # +1.0 == doubled
    dte = (date.fromisoformat(pos.expiration) - today).days

    if gain >= profit_target:
        return "take_profit", f"+{gain * 100:.0f}% (target +{profit_target * 100:.0f}%)"
    if gain <= -stop_loss:
        return "stop", f"{gain * 100:.0f}% (stop -{stop_loss * 100:.0f}%)"
    if dte <= expiry_exit_days:
        return "expiry", f"{dte}d to expiration"
    return "hold", f"{gain * 100:+.0f}% · {dte}d left"


def crossed_up50(pos: OptionPosition, current_premium: float) -> bool:
    """True the first time an open position is up >= 50% (for the milestone alert)."""
    if pos.up50_alerted or pos.premium_paid <= 0:
        return False
    return (current_premium - pos.premium_paid) / pos.premium_paid >= 0.50


def describe_bet(quote: OptionQuote, side_bias: str) -> str:
    return describe_bet_fields(quote.underlying, quote.type, quote.expiration)


def describe_bet_fields(underlying: str, opt_type: str, expiration: str) -> str:
    """Plain English: 'Betting Apple goes UP by Jul 10th'."""
    direction = "UP" if opt_type == "call" else "DOWN"
    try:
        d = date.fromisoformat(expiration)
        by = f"{d.strftime('%b')} {d.day}{_ordinal(d.day)}"
    except (ValueError, TypeError):
        by = expiration
    return f"Betting {friendly_name(underlying)} goes {direction} by {by}"


def _ordinal(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


# ============================================================================ #
# Network layer — Alpaca contracts/quotes + yfinance earnings
# ============================================================================ #
class OptionsStrategy:
    """Selects, sizes and evaluates long-option trades against live Alpaca data."""

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        *,
        paper: bool = True,
        dte_min: int = 30,
        dte_max: int = 45,
        risk_pct: float = 0.01,
        profit_target: float = 1.00,
        stop_loss: float = 0.50,
        max_positions: int = 3,
        skip_earnings: bool = True,
        expiry_exit_days: int = 1,
    ) -> None:
        from alpaca.trading.client import TradingClient
        from alpaca.data.historical.option import OptionHistoricalDataClient

        self._trading = TradingClient(api_key, secret_key, paper=paper)
        self._data = OptionHistoricalDataClient(api_key, secret_key)
        self.dte_min = dte_min
        self.dte_max = dte_max
        self.risk_pct = risk_pct
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.max_positions = max_positions
        self.skip_earnings = skip_earnings
        self.expiry_exit_days = expiry_exit_days
        self._earnings_cache: dict[str, Optional[date]] = {}

    # ------------------------------------------------------------------ #
    # Selection + sizing
    # ------------------------------------------------------------------ #
    def plan_trade(
        self, underlying: str, side: str, underlying_price: float, equity: float,
        score: float,
    ) -> Optional[OptionTradePlan]:
        """Build a sized option plan for a signal, or None if none is viable.

        ``side`` is the stock signal's bias: "long" -> call, "short" -> put.
        Applies the DTE window, ATM selection, earnings veto, and 1% sizing.
        """
        if "/" in underlying:
            logger.info("%s: options not supported for crypto — skipping", underlying)
            return None
        opt_type = "call" if side == "long" else "put"
        side_bias = "up" if opt_type == "call" else "down"

        quotes = self._list_priced_contracts(underlying, opt_type)
        if not quotes:
            logger.info("%s: no priced %s contracts in %d-%dd window",
                        underlying, opt_type, self.dte_min, self.dte_max)
            return None

        atm = select_atm(quotes, underlying_price)
        if atm is None:
            return None

        # Earnings veto: never hold through earnings before the contract expires.
        if self.skip_earnings:
            ed = self._earnings_before(underlying, date.fromisoformat(atm.expiration))
            if ed is not None:
                logger.info("%s: skipping option — earnings %s before expiry %s",
                            underlying, ed, atm.expiration)
                return None

        contracts = size_contracts(atm.premium, equity, self.risk_pct)
        if contracts < 1:
            logger.info("%s: ATM %s premium $%.2f too expensive for 1%% budget ($%.0f)",
                        underlying, opt_type, atm.contract_cost, equity * self.risk_pct)
            return None

        cost = atm.contract_cost * contracts
        return OptionTradePlan(
            quote=atm, contracts=contracts, side_bias=side_bias, score=score,
            cost=cost,
            target_premium=round(atm.premium * (1 + self.profit_target), 2),
            stop_premium=round(atm.premium * (1 - self.stop_loss), 2),
            risk_dollars=cost,   # long option max loss == premium paid
        )

    def _list_priced_contracts(self, underlying: str, opt_type: str) -> list[OptionQuote]:
        """Fetch active contracts in the DTE window and attach live premiums."""
        from alpaca.trading.requests import GetOptionContractsRequest
        from alpaca.trading.enums import ContractType, AssetStatus

        lo = date.today() + timedelta(days=self.dte_min)
        hi = date.today() + timedelta(days=self.dte_max)
        ctype = ContractType.CALL if opt_type == "call" else ContractType.PUT
        try:
            req = GetOptionContractsRequest(
                underlying_symbols=[underlying], status=AssetStatus.ACTIVE,
                type=ctype, expiration_date_gte=lo, expiration_date_lte=hi, limit=500,
            )
            contracts = self._trading.get_option_contracts(req).option_contracts
        except Exception:
            logger.exception("Failed to list option contracts for %s", underlying)
            return []
        if not contracts:
            return []

        # Restrict to the single nearest expiration in-window (one clean chain).
        exps = sorted({c.expiration_date for c in contracts})
        target_exp = exps[0]
        chain = [c for c in contracts if c.expiration_date == target_exp]
        symbols = [c.symbol for c in chain]
        prices = self._premiums(symbols)

        out: list[OptionQuote] = []
        for c in chain:
            px = prices.get(c.symbol)
            if not px:
                continue
            premium, bid, ask, delta, iv = px
            out.append(OptionQuote(
                symbol=c.symbol, underlying=underlying, type=opt_type,
                strike=float(c.strike_price),
                expiration=str(c.expiration_date), premium=premium,
                bid=bid, ask=ask, delta=delta, iv=iv,
            ))
        return out

    def _premiums(self, symbols: list[str]) -> dict[str, tuple]:
        """Map OCC symbol -> (mid_premium, bid, ask, delta, iv) from snapshots.

        Premium is the bid/ask mid when both sides quote, else the last trade.
        """
        from alpaca.data.requests import OptionSnapshotRequest

        out: dict[str, tuple] = {}
        # Snapshots accept batches; chunk to stay well under URL limits.
        for i in range(0, len(symbols), 100):
            batch = symbols[i:i + 100]
            try:
                snaps = self._data.get_option_snapshot(
                    OptionSnapshotRequest(symbol_or_symbols=batch))
            except Exception:
                logger.exception("Option snapshot batch failed")
                continue
            for sym, snap in snaps.items():
                if snap is None:
                    continue
                q = getattr(snap, "latest_quote", None)
                t = getattr(snap, "latest_trade", None)
                bid = float(getattr(q, "bid_price", 0) or 0) if q else 0.0
                ask = float(getattr(q, "ask_price", 0) or 0) if q else 0.0
                if bid > 0 and ask > 0:
                    premium = round((bid + ask) / 2, 2)
                elif t and getattr(t, "price", 0):
                    premium = round(float(t.price), 2)
                else:
                    continue
                g = getattr(snap, "greeks", None)
                delta = float(getattr(g, "delta", 0)) if g else None
                iv = getattr(snap, "implied_volatility", None)
                out[sym] = (premium, bid, ask, delta,
                            float(iv) if iv is not None else None)
        return out

    def current_premium(self, option_symbol: str) -> Optional[float]:
        """Live mid premium for one OCC symbol (used to re-price open positions)."""
        return (self._premiums([option_symbol]).get(option_symbol) or (None,))[0]

    # ------------------------------------------------------------------ #
    # Earnings calendar (yfinance) — never hold through earnings
    # ------------------------------------------------------------------ #
    def _earnings_before(self, symbol: str, expiry: date) -> Optional[date]:
        """Return the next earnings date if it falls on/before ``expiry``, else None.

        Best-effort: on any data/network failure we log and return None (allow the
        trade) rather than blocking on a flaky calendar.
        """
        nxt = self._next_earnings(symbol)
        if nxt is None:
            return None
        return nxt if date.today() <= nxt <= expiry else None

    def _next_earnings(self, symbol: str) -> Optional[date]:
        if symbol in self._earnings_cache:
            return self._earnings_cache[symbol]
        nxt: Optional[date] = None
        try:
            import yfinance as yf
            t = yf.Ticker(symbol)
            df = t.get_earnings_dates(limit=12)
            if df is not None and not df.empty:
                today = datetime.now(timezone.utc)
                future = [d for d in df.index.to_pydatetime()
                          if d.replace(tzinfo=d.tzinfo or timezone.utc) >= today]
                if future:
                    nxt = min(future).date()
        except Exception:
            logger.info("%s: earnings lookup failed — allowing trade", symbol)
        self._earnings_cache[symbol] = nxt
        return nxt


# ============================================================================ #
# Persistence — open option positions survive a restart
# ============================================================================ #
class OptionPositionStore:
    def __init__(self, log_dir: str = "logs"):
        self.path = os.path.join(log_dir, "option_positions.json")
        os.makedirs(log_dir, exist_ok=True)

    def save(self, positions: dict[str, OptionPosition]) -> None:
        try:
            data = {sym: p.to_dict() for sym, p in positions.items()
                    if p.status == "open"}
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, default=str)
            os.replace(tmp, self.path)
        except Exception:
            logger.exception("OptionPositionStore.save failed")

    def load(self) -> dict[str, OptionPosition]:
        try:
            with open(self.path) as f:
                data = json.load(f)
            return {sym: OptionPosition.from_dict(d) for sym, d in data.items()}
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
        except Exception:
            logger.exception("OptionPositionStore.load failed")
            return {}
