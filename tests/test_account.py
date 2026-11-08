import pandas as pd
import pytest

from tradelab.account import PaperAccount
from tradelab.costs import FREE, CostModel

T = pd.Timestamp("2024-01-01", tz="UTC")


def acct(cash=1000.0, costs=FREE):
    return PaperAccount(cash=cash, costs=costs)


def test_opening_with_no_costs_leaves_equity_unchanged():
    a = acct()
    a.open_position("long", 1.0, 100.0, 2, T)
    assert a.equity(100.0) == pytest.approx(1000.0)
    assert a.margin_used() == pytest.approx(50.0)           # 100 / levier 2
    assert a.free_margin(100.0) == pytest.approx(950.0)


def test_long_pnl_and_close():
    a = acct()
    a.open_position("long", 2.0, 100.0, 1, T)
    assert a.equity(110.0) == pytest.approx(1020.0)
    fill = a.close_position("long", None, 110.0, T)
    assert fill.pnl == pytest.approx(20.0) and a.long is None and a.cash == pytest.approx(1020.0)
    assert a.margin_used() == 0


def test_short_gains_when_price_falls():
    a = acct()
    a.open_position("short", 2.0, 100.0, 1, T)
    assert a.equity(90.0) == pytest.approx(1020.0)
    a.close_position("short", None, 90.0, T)
    assert a.cash == pytest.approx(1020.0)


def test_fees_and_slippage_are_charged_on_notional():
    c = CostModel(taker_fee=0.01, slippage=0.01, rollover_per_4h=0)
    a = acct(costs=c)
    fill = a.open_position("long", 1.0, 100.0, 1, T)
    assert fill.price == pytest.approx(101.0) and fill.fee == pytest.approx(1.01)
    assert a.cash == pytest.approx(1000 - 1.01)
    out = a.close_position("long", None, 100.0, T)
    assert out.price == pytest.approx(99.0)
    # aller-retour au même prix médian : on perd glissement ×2 et frais ×2
    assert a.cash == pytest.approx(1000 - 1.01 - 2.0 - 0.99)


def test_open_is_rejected_when_free_margin_is_insufficient():
    a = acct(cash=10.0)
    assert a.open_position("long", 1.0, 100.0, 5, T) is None      # il faudrait 20 de marge
    assert a.rejected == 1 and a.long is None and a.cash == 10.0
    assert a.open_position("long", 0.5, 100.0, 5, T) is not None   # 10 de marge : juste suffisant


def test_adding_to_position_averages_entry_and_accumulates_margin():
    a = acct()
    a.open_position("long", 1.0, 100.0, 2, T)
    a.open_position("long", 3.0, 120.0, 4, T)
    assert a.long.qty == pytest.approx(4.0)
    assert a.long.entry == pytest.approx((100 + 360) / 4)
    assert a.long.margin == pytest.approx(50 + 90)


def test_partial_close_releases_margin_proportionally():
    a = acct()
    a.open_position("long", 4.0, 100.0, 2, T)                     # marge 200
    a.close_position("long", 1.0, 100.0, T)
    assert a.long.qty == pytest.approx(3.0) and a.long.margin == pytest.approx(150.0)


def test_close_more_than_held_closes_only_what_exists():
    a = acct()
    a.open_position("long", 1.0, 100.0, 1, T)
    fill = a.close_position("long", 5.0, 100.0, T)
    assert fill.qty == 1.0 and a.long is None
    assert a.close_position("long", 1.0, 100.0, T) is None         # plus rien à fermer


def test_long_and_short_can_coexist_in_hedge_mode():
    a = acct()
    a.open_position("long", 1.0, 100.0, 2, T)
    a.open_position("short", 1.0, 100.0, 2, T)
    assert a.net_exposure(100.0) == pytest.approx(0.0)
    assert a.gross_exposure(100.0) == pytest.approx(0.2)
    assert a.margin_used() == pytest.approx(100.0)


def test_invalid_orders_raise():
    a = acct()
    with pytest.raises(ValueError):
        a.open_position("long", 0.0, 100.0, 2, T)
    with pytest.raises(ValueError):
        a.open_position("long", 1.0, 100.0, 0, T)


def test_rollover_is_charged_on_borrowed_amount_only():
    c = CostModel(taker_fee=0, slippage=0, rollover_per_4h=0.001)
    a = acct(costs=c)
    a.open_position("long", 10.0, 100.0, 2, T)                    # notionnel 1000, marge 500, emprunt 500
    paid = a.accrue_rollover(4.0)
    assert paid == pytest.approx(0.5) and a.cash == pytest.approx(999.5) and a.rollover_paid == pytest.approx(0.5)
    b = acct(costs=c)
    b.open_position("long", 10.0, 100.0, 1, T)                    # sans levier : rien d'emprunté
    assert b.accrue_rollover(24.0) == 0.0


def test_margin_level():
    a = acct()
    assert a.margin_level(100.0) == float("inf")
    a.open_position("long", 10.0, 100.0, 2, T)                    # marge 500 ; équité 1000
    assert a.margin_level(100.0) == pytest.approx(2.0)
    assert a.margin_level(70.0) == pytest.approx((1000 - 300) / 500)


def test_liquidation_triggers_at_the_worst_price_of_the_bar():
    c = CostModel(taker_fee=0, slippage=0, rollover_per_4h=0, liquidation_fee=0.02, liquidation_level=0.4)
    a = acct(cash=100.0, costs=c)
    a.open_position("long", 2.0, 100.0, 2, T)                     # notionnel 200, marge 100
    assert not a.check_liquidation(low=80.0, high=101.0, ts=T)    # équité 60 > 0,4 × 100 = 40
    assert a.check_liquidation(low=70.0, high=101.0, ts=T)        # équité 40 : seuil atteint
    assert a.long is None and a.liquidations == 1
    # perte 60 + frais de liquidation 2 % × 140 = 2,8
    assert a.cash == pytest.approx(100 - 60 - 2.8)


def test_short_is_liquidated_on_the_high():
    c = CostModel(taker_fee=0, slippage=0, rollover_per_4h=0, liquidation_fee=0.0, liquidation_level=0.4)
    a = acct(cash=100.0, costs=c)
    a.open_position("short", 2.0, 100.0, 2, T)
    assert not a.check_liquidation(low=90.0, high=120.0, ts=T)
    assert a.check_liquidation(low=90.0, high=130.0, ts=T)


def test_liquidation_can_ruin_the_account_and_floors_cash_at_zero():
    c = CostModel(taker_fee=0, slippage=0, rollover_per_4h=0, liquidation_fee=0.5, liquidation_level=0.4)
    a = acct(cash=100.0, costs=c)
    a.open_position("long", 2.0, 100.0, 2, T)
    a.check_liquidation(low=60.0, high=100.0, ts=T)
    assert a.cash == 0.0 and a.ruined


def test_no_position_no_liquidation():
    assert not acct().check_liquidation(1.0, 1000.0, T)
