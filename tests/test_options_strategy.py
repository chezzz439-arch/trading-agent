"""Unit tests for the pure options decision logic (no network)."""

from __future__ import annotations

from datetime import date, timedelta

from src.signals.options_strategy import (
    OptionPosition,
    OptionQuote,
    crossed_up50,
    describe_bet_fields,
    exit_decision,
    select_atm,
    size_contracts,
)


def _q(strike, premium, exp_days=37, typ="call"):
    exp = (date.today() + timedelta(days=exp_days)).isoformat()
    return OptionQuote(symbol=f"X{strike}", underlying="AAPL", type=typ,
                       strike=strike, expiration=exp, premium=premium)


# --- ATM selection --------------------------------------------------------- #
def test_select_atm_picks_closest_strike():
    quotes = [_q(190, 5.0), _q(200, 3.2), _q(205, 2.1), _q(210, 1.5)]
    atm = select_atm(quotes, underlying_price=201.0)
    assert atm.strike == 200


def test_select_atm_skips_unpriced():
    quotes = [_q(200, 0.0), _q(205, 2.0)]
    atm = select_atm(quotes, underlying_price=201.0)
    assert atm.strike == 205   # the 200 has no premium, so 205 wins


def test_select_atm_empty():
    assert select_atm([], 100.0) is None


# --- sizing ---------------------------------------------------------------- #
def test_size_contracts_within_budget():
    # 1% of 100k = $1,000 budget; $3.20 premium = $320/contract -> 3 contracts.
    assert size_contracts(premium=3.20, equity=100_000, risk_pct=0.01) == 3


def test_size_contracts_too_expensive_returns_zero():
    # $12 premium = $1,200/contract > $1,000 budget.
    assert size_contracts(premium=12.0, equity=100_000, risk_pct=0.01) == 0


def test_size_contracts_guards_bad_input():
    assert size_contracts(premium=0, equity=100_000, risk_pct=0.01) == 0
    assert size_contracts(premium=3.0, equity=0, risk_pct=0.01) == 0


# --- exit decision --------------------------------------------------------- #
def _pos(paid=3.0, exp_days=30):
    exp = (date.today() + timedelta(days=exp_days)).isoformat()
    return OptionPosition(symbol="X", underlying="AAPL", type="call", strike=200,
                          expiration=exp, contracts=3, premium_paid=paid,
                          cost_basis=paid * 100 * 3, side_bias="up",
                          target_premium=paid * 2, stop_premium=paid * 0.5)


def test_exit_take_profit_at_double():
    action, _ = exit_decision(_pos(3.0), current_premium=6.0,
                              profit_target=1.0, stop_loss=0.5)
    assert action == "take_profit"


def test_exit_stop_at_half():
    action, _ = exit_decision(_pos(3.0), current_premium=1.5,
                              profit_target=1.0, stop_loss=0.5)
    assert action == "stop"


def test_exit_hold_in_between():
    action, _ = exit_decision(_pos(3.0), current_premium=3.6,
                              profit_target=1.0, stop_loss=0.5)
    assert action == "hold"


def test_exit_expiry_when_near_expiration():
    action, _ = exit_decision(_pos(3.0, exp_days=1), current_premium=3.3,
                              profit_target=1.0, stop_loss=0.5, expiry_exit_days=1)
    assert action == "expiry"


def test_take_profit_beats_expiry_when_both_true():
    # Doubled AND expiring -> profit wins (we cash the win).
    action, _ = exit_decision(_pos(3.0, exp_days=0), current_premium=6.0,
                              profit_target=1.0, stop_loss=0.5, expiry_exit_days=1)
    assert action == "take_profit"


# --- up-50 milestone ------------------------------------------------------- #
def test_crossed_up50_true_once():
    pos = _pos(3.0)
    assert crossed_up50(pos, current_premium=4.6) is True   # +53%
    pos.up50_alerted = True
    assert crossed_up50(pos, current_premium=5.0) is False  # already alerted


def test_crossed_up50_false_below():
    assert crossed_up50(_pos(3.0), current_premium=4.0) is False  # +33%


# --- plain English --------------------------------------------------------- #
def test_describe_call_is_up():
    txt = describe_bet_fields("AAPL", "call", (date(2026, 12, 20)).isoformat())
    assert txt == "Betting Apple goes UP by Dec 20th"


def test_describe_put_is_down():
    txt = describe_bet_fields("TSLA", "put", (date(2026, 7, 1)).isoformat())
    assert txt == "Betting Tesla goes DOWN by Jul 1st"
