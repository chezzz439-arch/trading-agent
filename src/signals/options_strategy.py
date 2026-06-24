"""Options strategy — turn a strong stock signal into a long call/put.

When the master scorer produces a high-conviction signal the agent can express
it with a defined-risk **long option** instead of the stock:

* strong **LONG**  (score >= gate)  ->  buy a **call**
* strong **SHORT** (score >= gate)  ->  buy a **put**

Design (mirrors ``position_manager``): the **pure decision logic** — contract
selection, premium sizing, the entry-quality gates, and the take-profit / stop /
expiry evaluation — is separated from the **network layer** (Alpaca contract
listing + option market data, and the yfinance earnings calendar) so the core is
fully unit-testable offline and behaves identically in a paper test and the live
loop.

Entry-quality gates (rebuilt 2026-06-23 — all configurable in ``settings``):

1. Post-open delay   — no entries within OPTIONS_OPEN_DELAY_MIN of the open
                       (opening IV is inflated; it settles → "IV crush").
2. IV-rank gate      — buy premium only when IV is not expensive
                       (IV rank <= OPTIONS_IV_RANK_MAX; <= PREFERRED is cheap).
3. Spread gate       — skip contracts whose bid-ask spread exceeds
                       OPTIONS_MAX_SPREAD_PCT of the mid (too illiquid). Entries
                       are LIMIT orders at the mid (caller), never market orders.
4. Greeks selection  — pick a contract with |delta| in [DELTA_MIN, DELTA_MAX],
                       30–45 DTE, and daily theta decay <= OPTIONS_MAX_THETA_PCT.
5. Fill validation   — caller only records a position after a confirmed, sane
                       fill (no phantom positions from unfilled paper orders).

Long options only — no shorting/writing — so max loss is always the premium
paid, which is what the 1% sizing budgets for.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict, dataclass
from datetime import date, datetime, time as dtime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# US equity options regular-session open, in US/Eastern.
MARKET_OPEN_ET = dtime(9, 30)

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


def is_option_asset(asset_class) -> bool:
    """True if an Alpaca ``asset_class`` denotes a US option.

    Robust to both forms the SDK returns: the ``AssetClass`` enum (whose
    ``str()`` is ``"AssetClass.US_OPTION"``) and a raw ``"us_option"`` string.
    The naive ``str(ac).endswith("us_option")`` is a TRAP — it is False for the
    enum because str() yields "AssetClass.US_OPTION", so option positions were
    silently invisible. Match case-insensitively on substring instead.
    """
    return "us_option" in str(asset_class).lower()


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
    theta: Optional[float] = None
    vega: Optional[float] = None
    gamma: Optional[float] = None
    iv: Optional[float] = None

    @property
    def dte(self) -> int:
        return (date.fromisoformat(self.expiration) - date.today()).days

    @property
    def contract_cost(self) -> float:
        return self.premium * 100.0


@dataclass
class OptionTradePlan:
    """A fully-sized, ready-to-place option order with its exit levels + diagnostics."""
    quote: OptionQuote
    contracts: int
    side_bias: str          # "up" (call) | "down" (put)
    score: float
    cost: float             # total premium outlay = premium*100*contracts
    target_premium: float   # sell here for the +100% profit target
    stop_premium: float     # cut here for the -50% stop
    risk_dollars: float     # max loss = the premium paid (defined risk)
    limit_price: float      # entry LIMIT price (mid + buffer toward ask)
    # diagnostics surfaced into the trade reasoning / dashboard
    iv_rank: Optional[float] = None
    iv_rank_method: str = ""      # "history" | "hv_proxy" | "none"
    spread_pct: Optional[float] = None

    @property
    def description(self) -> str:
        return describe_bet(self.quote, self.side_bias)

    @property
    def greeks_line(self) -> str:
        q = self.quote
        def f(x):
            return f"{x:+.3f}" if isinstance(x, (int, float)) else "—"
        ivr = f"{self.iv_rank:.0f}" if self.iv_rank is not None else "—"
        return (f"Δ{f(q.delta)} θ{f(q.theta)} v{f(q.vega)} "
                f"IV{(q.iv*100):.0f}% IVR{ivr}({self.iv_rank_method}) "
                f"DTE{q.dte} spread{(self.spread_pct or 0)*100:.0f}%")


@dataclass
class OptionPosition:
    """Persisted state for one open option position (current value comes live)."""
    symbol: str             # OCC symbol
    underlying: str
    type: str               # call | put
    strike: float
    expiration: str         # ISO date
    contracts: int
    premium_paid: float     # per-share entry premium (the actual fill price)
    cost_basis: float       # premium_paid * 100 * contracts
    side_bias: str          # up | down
    score: float = 0.0
    target_premium: float = 0.0
    stop_premium: float = 0.0
    entry_time: str = ""
    # alert/lifecycle flags so we notify each milestone exactly once
    up50_alerted: bool = False
    status: str = "open"    # open | closed
    # fill tracking: True once we see this position confirmed in broker positions.
    # An option that disappears before broker_confirmed was ever set was an
    # unfilled order — don't count it as a real loss toward consecutive losses.
    broker_confirmed: bool = False
    missing_scans: int = 0  # consecutive scans where broker position was absent

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
    """Pick the at-the-money contract: strike closest to the current price.

    Retained for backward-compatibility / fallback; the live path prefers
    ``select_by_greeks``. Ties break toward the lower strike.
    """
    priced = [q for q in quotes if q.premium and q.premium > 0]
    if not priced:
        return None
    return min(priced, key=lambda q: (abs(q.strike - underlying_price), q.strike))


def spread_pct(bid: float, ask: float) -> Optional[float]:
    """Bid-ask spread as a fraction of the mid. None if not two-sided/finite."""
    if bid is None or ask is None:
        return None
    if not (math.isfinite(bid) and math.isfinite(ask)):
        return None
    if bid <= 0 or ask <= 0 or ask < bid:
        return None
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return None
    return (ask - bid) / mid


def passes_spread(quote: OptionQuote, max_spread_pct: float) -> bool:
    """True if the contract is liquid enough (tight two-sided market)."""
    sp = spread_pct(quote.bid, quote.ask)
    return sp is not None and sp <= max_spread_pct


def theta_decay_pct(theta: Optional[float], premium: float) -> Optional[float]:
    """Daily theta burn as a fraction of premium (e.g. 0.02 == 2%/day). None if unknown.

    Guards finiteness so a NaN/inf theta or premium can't slip past the caller's
    ``> max`` comparison (NaN > x is False, which would wrongly read as "passes").
    """
    if theta is None or premium is None:
        return None
    if not (math.isfinite(theta) and math.isfinite(premium)) or premium <= 0:
        return None
    return abs(theta) / premium


def select_by_greeks(
    quotes: list[OptionQuote],
    *,
    delta_min: float,
    delta_max: float,
    max_spread_pct: float,
    max_theta_pct: float,
) -> Optional[OptionQuote]:
    """Pick the best contract by Greeks rather than raw ATM distance.

    Eligible contracts must have:
      * |delta| within [delta_min, delta_max]   (directional, not over-paying),
      * a two-sided spread <= max_spread_pct     (liquid),
      * daily theta decay <= max_theta_pct       (not bleeding too fast),
      * a real premium and IV/greeks present.

    Among the eligible set, prefer the contract whose |delta| is closest to the
    band midpoint (a balanced ~0.6-delta directional bet), then the tighter
    spread as a tie-breaker. Returns None if nothing qualifies (caller skips).
    """
    target = (delta_min + delta_max) / 2.0
    eligible: list[tuple[float, float, OptionQuote]] = []
    for q in quotes:
        # Finiteness guards first — a NaN premium/delta otherwise sneaks through
        # the band/zero comparisons (NaN inequalities are all False).
        if q.premium is None or not math.isfinite(q.premium) or q.premium <= 0:
            continue
        if q.delta is None or not math.isfinite(q.delta):
            continue
        adelta = abs(q.delta)
        if not (delta_min <= adelta <= delta_max):
            continue
        if not passes_spread(q, max_spread_pct):
            continue
        td = theta_decay_pct(q.theta, q.premium)
        if td is None or td > max_theta_pct:
            continue
        sp = spread_pct(q.bid, q.ask) or 0.0
        eligible.append((abs(adelta - target), sp, q))
    if not eligible:
        return None
    eligible.sort(key=lambda t: (t[0], t[1]))
    return eligible[0][2]


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


def limit_entry_price(bid: float, ask: float, buffer_pct: float) -> Optional[float]:
    """LIMIT price for a buy: mid nudged ``buffer_pct`` of the spread toward the ask.

    buffer_pct=0 → exact mid; 1.0 → the ask. Never exceeds the ask. Rounded to a
    cent (options trade in $0.01 increments for prices under $3, but a cent is a
    safe, broadly-accepted tick). None if not two-sided.
    """
    if bid is None or ask is None or bid <= 0 or ask <= 0 or ask < bid:
        return None
    mid = (bid + ask) / 2.0
    px = mid + (ask - mid) * max(0.0, min(1.0, buffer_pct))
    px = min(px, ask)
    return round(px, 2)


# --- Post-open entry delay (IV-crush timing) ------------------------------- #
def minutes_since_open(now: datetime, market_open: dtime = MARKET_OPEN_ET) -> float:
    """Minutes elapsed since the 9:30 ET bell today. Negative before the open.

    Converts ``now`` to America/New_York internally so the result is correct
    regardless of the input zone (a naive datetime is assumed to already be ET).
    The 9:30 open is a constant wall-clock time NYSE keeps through DST/half-days,
    so anchoring to the ET local date is correct.
    """
    from zoneinfo import ZoneInfo
    et = ZoneInfo("America/New_York")
    now_et = now if now.tzinfo is None else now.astimezone(et)
    open_dt = datetime.combine(now_et.date(), market_open, tzinfo=now_et.tzinfo)
    return (now_et - open_dt).total_seconds() / 60.0


def within_open_delay(now_et: datetime, delay_min: int,
                      market_open: dtime = MARKET_OPEN_ET) -> bool:
    """True if we are inside the no-entry window [open, open + delay_min).

    Before the open returns False (the market gate elsewhere blocks those);
    this guard specifically covers the inflated-IV minutes right after the bell.
    """
    m = minutes_since_open(now_et, market_open)
    return 0.0 <= m < float(max(0, delay_min))


# --- IV rank (true history + realized-vol bootstrap proxy) ------------------ #
def iv_rank(current_iv: Optional[float], iv_history: list[float],
            min_samples: int) -> Optional[float]:
    """True IV rank: percentile of current IV within its trailing history (0-100).

    ``(current - min) / (max - min) * 100`` over the stored IV observations.
    Returns None when there isn't enough history (caller falls back to a proxy).
    """
    if current_iv is None or not math.isfinite(current_iv) or not iv_history:
        return None
    hist = [float(x) for x in iv_history
            if x is not None and math.isfinite(x) and x > 0]
    if len(hist) < max(2, min_samples):
        return None
    lo, hi = min(hist), max(hist)
    if hi <= lo:
        # Flat history → genuinely indeterminate. Return None so the caller can
        # fail CLOSED (skip) rather than silently treating a persistently-rich
        # name as mid-range and letting it trade.
        return None
    rank = (current_iv - lo) / (hi - lo) * 100.0
    return max(0.0, min(100.0, rank))


def realized_vol_series(closes: list[float], window: int = 30) -> list[float]:
    """Rolling annualized realized volatility from a daily close series.

    For each trailing ``window`` of daily log returns: std * sqrt(252). Used to
    bootstrap an IV-rank proxy before we have a real IV history.
    """
    closes = [float(c) for c in closes if c is not None and c > 0]
    if len(closes) < window + 1:
        return []
    rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
    out: list[float] = []
    for i in range(window, len(rets) + 1):
        w = rets[i - window:i]
        n = len(w)
        if n < 2:
            continue
        mean = sum(w) / n
        var = sum((r - mean) ** 2 for r in w) / (n - 1)
        out.append(math.sqrt(var) * math.sqrt(252.0))
    return out


def iv_rank_proxy_from_hv(current_iv: Optional[float], closes: list[float],
                          window: int = 30) -> Optional[float]:
    """Bootstrap IV-rank proxy: rank current IV within the underlying's trailing
    realized-vol range over the available (~1y) daily closes.

    This is a *proxy*, not a true IV rank — Alpaca exposes no historical IV. It
    answers "is implied vol high or low versus how much this name actually
    moves?", which is the property the IV-rank gate cares about. Clearly labelled
    as ``hv_proxy`` wherever it is used.
    """
    if current_iv is None or not math.isfinite(current_iv):
        return None
    hv = realized_vol_series(closes, window)
    if len(hv) < 2:
        return None
    lo, hi = min(hv), max(hv)
    if hi <= lo:
        return None  # flat HV → indeterminate; caller fails closed
    rank = (current_iv - lo) / (hi - lo) * 100.0
    return max(0.0, min(100.0, rank))


def resolve_iv_rank(current_iv: Optional[float], iv_history: list[float],
                    daily_closes: list[float], min_samples: int,
                    hv_window: int = 30) -> tuple[Optional[float], str]:
    """Return ``(iv_rank, method)`` preferring true history, else the HV proxy.

    method ∈ {"history", "hv_proxy", "none"}.
    """
    r = iv_rank(current_iv, iv_history, min_samples)
    if r is not None:
        return r, "history"
    r = iv_rank_proxy_from_hv(current_iv, daily_closes, hv_window)
    if r is not None:
        return r, "hv_proxy"
    return None, "none"


def fill_price_is_sane(fill_price: Optional[float], bid: float, ask: float) -> bool:
    """Guard against phantom/garbage fills before recording a position.

    A real long-option buy should fill at a positive price near the quoted
    market: within [0.5*bid, 1.5*ask]. Rejects 0, None, NaN and absurd prints.
    """
    if fill_price is None:
        return False
    try:
        fp = float(fill_price)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(fp) or fp <= 0:
        return False
    lo = 0.5 * bid if bid and bid > 0 else 0.0
    hi = 1.5 * ask if ask and ask > 0 else fp * 1.5
    # If we have no usable quote, accept any positive finite price.
    if lo <= 0 and (not ask or ask <= 0):
        return True
    return lo <= fp <= hi


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
# IV history persistence — builds a real per-underlying IV series over time
# ============================================================================ #
class IVHistoryStore:
    """Append-only daily ATM-IV observations per underlying (trailing ~1y).

    One sample per underlying per day. Over time this accumulates the real IV
    series the true ``iv_rank`` needs; until then callers use the HV proxy.
    """

    def __init__(self, log_dir: str = "logs", max_samples: int = 365):
        self.path = os.path.join(log_dir, "iv_history.json")
        self.max_samples = max_samples
        os.makedirs(log_dir, exist_ok=True)
        self._data: dict[str, list[dict]] = self._load()

    def _load(self) -> dict[str, list[dict]]:
        try:
            with open(self.path) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
        except Exception:
            logger.exception("IVHistoryStore load failed")
            return {}

    def _save(self) -> None:
        try:
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self._data, f)
            os.replace(tmp, self.path)
        except Exception:
            logger.exception("IVHistoryStore save failed")

    def record(self, underlying: str, iv: float, day: Optional[date] = None) -> None:
        """Store one IV observation for ``underlying`` (one clean sample per day).

        Validates the IV is finite and in a sane range (0 < iv < 10 == 1000%) so
        a garbage snapshot print can't poison the trailing series. Keeps the
        FIRST clean sample of the day (a late-day bad print won't overwrite it).
        """
        if iv is None or not math.isfinite(iv) or not (0.0 < iv < 10.0):
            return
        day = day or date.today()
        iso = day.isoformat()
        series = self._data.setdefault(underlying, [])
        if series and series[-1].get("date") == iso:
            return  # already have today's sample — keep the first clean one
        series.append({"date": iso, "iv": float(iv)})
        if len(series) > self.max_samples:
            del series[: len(series) - self.max_samples]
        self._save()

    def history(self, underlying: str) -> list[float]:
        return [float(d["iv"]) for d in self._data.get(underlying, [])
                if d.get("iv") is not None]


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
        # rebuilt entry-quality gates
        delta_min: float = 0.55,
        delta_max: float = 0.70,
        max_spread_pct: float = 0.15,
        max_theta_pct: float = 0.02,
        iv_rank_max: float = 50.0,
        iv_rank_preferred: float = 30.0,
        iv_history_min_samples: int = 20,
        limit_buffer_pct: float = 0.02,
        log_dir: str = "logs",
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
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.max_spread_pct = max_spread_pct
        self.max_theta_pct = max_theta_pct
        self.iv_rank_max = iv_rank_max
        self.iv_rank_preferred = iv_rank_preferred
        self.iv_history_min_samples = iv_history_min_samples
        self.limit_buffer_pct = limit_buffer_pct
        self.iv_store = IVHistoryStore(log_dir=log_dir)
        self._earnings_cache: dict[str, Optional[date]] = {}

    # ------------------------------------------------------------------ #
    # Selection + sizing
    # ------------------------------------------------------------------ #
    def plan_trade(
        self, underlying: str, side: str, underlying_price: float, equity: float,
        score: float, *, daily_closes: Optional[list[float]] = None,
    ) -> Optional[OptionTradePlan]:
        """Build a sized, gated option plan for a signal, or None if not viable.

        ``side`` is the stock signal's bias: "long" -> call, "short" -> put.
        Applies, in order: DTE window, Greeks selection (delta band + spread +
        theta), earnings veto, IV-rank gate, sizing, and a LIMIT entry price.
        The post-open-delay gate is enforced by the caller (it owns the clock).
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

        # Fix 4 — Greeks-based selection (delta band + spread + theta gates).
        pick = select_by_greeks(
            quotes, delta_min=self.delta_min, delta_max=self.delta_max,
            max_spread_pct=self.max_spread_pct, max_theta_pct=self.max_theta_pct)
        if pick is None:
            logger.info("%s: no %s contract in delta %.2f-%.2f with spread<=%.0f%% "
                        "and theta<=%.0f%%/day — skipping",
                        underlying, opt_type, self.delta_min, self.delta_max,
                        self.max_spread_pct * 100, self.max_theta_pct * 100)
            return None

        # Record the chosen contract's IV into the rolling history (builds the
        # real series for future true IV-rank), then resolve IV rank.
        if pick.iv is not None and pick.iv > 0:
            self.iv_store.record(underlying, pick.iv)
        ivr, method = resolve_iv_rank(
            pick.iv, self.iv_store.history(underlying), daily_closes or [],
            self.iv_history_min_samples)

        # Fix 2 — IV-rank gate: don't buy premium when implied vol is expensive.
        # Fails CLOSED: if IV rank can't be determined (no history AND no usable
        # HV proxy), skip rather than buy blind — options touch real risk.
        if ivr is None:
            logger.info("%s: IV rank indeterminate (%s) — failing closed, skipping",
                        underlying, method)
            return None
        if ivr > self.iv_rank_max:
            logger.info("%s: IV too expensive — IV rank %.0f > %.0f (%s) — skipping",
                        underlying, ivr, self.iv_rank_max, method)
            return None
        if ivr > self.iv_rank_preferred:
            logger.info("%s: IV rank %.0f (%s) elevated but <= %.0f — allowed",
                        underlying, ivr, method, self.iv_rank_max)

        # Earnings veto: never hold through earnings before the contract expires.
        if self.skip_earnings:
            ed = self._earnings_before(underlying, date.fromisoformat(pick.expiration))
            if ed is not None:
                logger.info("%s: skipping option — earnings %s before expiry %s",
                            underlying, ed, pick.expiration)
                return None

        contracts = size_contracts(pick.premium, equity, self.risk_pct)
        if contracts < 1:
            logger.info("%s: %s premium $%.2f too expensive for %.0f%% budget ($%.0f)",
                        underlying, opt_type, pick.contract_cost,
                        self.risk_pct * 100, equity * self.risk_pct)
            return None

        cost = pick.contract_cost * contracts
        limit_px = limit_entry_price(pick.bid, pick.ask, self.limit_buffer_pct)
        if limit_px is None:
            logger.info("%s: no two-sided market for %s — skipping", underlying, pick.symbol)
            return None
        return OptionTradePlan(
            quote=pick, contracts=contracts, side_bias=side_bias, score=score,
            cost=cost,
            target_premium=round(pick.premium * (1 + self.profit_target), 2),
            stop_premium=round(pick.premium * (1 - self.stop_loss), 2),
            risk_dollars=cost,   # long option max loss == premium paid
            limit_price=limit_px,
            iv_rank=ivr, iv_rank_method=method,
            spread_pct=spread_pct(pick.bid, pick.ask),
        )

    def _list_priced_contracts(self, underlying: str, opt_type: str) -> list[OptionQuote]:
        """Fetch active contracts in the DTE window and attach live premiums/greeks."""
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
            premium, bid, ask, delta, theta, vega, gamma, iv = px
            out.append(OptionQuote(
                symbol=c.symbol, underlying=underlying, type=opt_type,
                strike=float(c.strike_price),
                expiration=str(c.expiration_date), premium=premium,
                bid=bid, ask=ask, delta=delta, theta=theta, vega=vega,
                gamma=gamma, iv=iv,
            ))
        return out

    def _premiums(self, symbols: list[str]) -> dict[str, tuple]:
        """Map OCC symbol -> (mid, bid, ask, delta, theta, vega, gamma, iv).

        Premium is the bid/ask mid when both sides quote, else the last trade.
        """
        from alpaca.data.requests import OptionSnapshotRequest

        out: dict[str, tuple] = {}
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
                delta = float(getattr(g, "delta", 0)) if g and getattr(g, "delta", None) is not None else None
                theta = float(getattr(g, "theta", 0)) if g and getattr(g, "theta", None) is not None else None
                vega = float(getattr(g, "vega", 0)) if g and getattr(g, "vega", None) is not None else None
                gamma = float(getattr(g, "gamma", 0)) if g and getattr(g, "gamma", None) is not None else None
                iv = getattr(snap, "implied_volatility", None)
                out[sym] = (premium, bid, ask, delta, theta, vega, gamma,
                            float(iv) if iv is not None else None)
        return out

    def snapshot_quote(self, option_symbol: str) -> Optional[OptionQuote]:
        """Live single-contract quote (bid/ask/greeks) — used for fill checks."""
        px = self._premiums([option_symbol]).get(option_symbol)
        if not px:
            return None
        premium, bid, ask, delta, theta, vega, gamma, iv = px
        return OptionQuote(symbol=option_symbol, underlying="", type="",
                           strike=0.0, expiration="", premium=premium, bid=bid,
                           ask=ask, delta=delta, theta=theta, vega=vega,
                           gamma=gamma, iv=iv)

    def current_premium(self, option_symbol: str) -> Optional[float]:
        """Live mid premium for one OCC symbol (used to re-price open positions)."""
        px = self._premiums([option_symbol]).get(option_symbol)
        return px[0] if px else None

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
