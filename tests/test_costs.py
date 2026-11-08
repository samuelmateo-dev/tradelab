import pytest

from tradelab.costs import FREE, CostModel, legacy_reported_pnl, true_round_trip_pnl


def test_fee_is_proportional_to_notional():
    c = CostModel(taker_fee=0.0026)
    assert c.fee(1000) == pytest.approx(2.6)
    assert c.fee(-1000) == pytest.approx(2.6)


def test_slippage_always_hurts():
    c = CostModel(slippage=0.001)
    assert c.buy_price(100) == pytest.approx(100.1)
    assert c.sell_price(100) == pytest.approx(99.9)


def test_rollover_only_on_borrowed_amount():
    c = CostModel(rollover_per_4h=0.0002)
    assert c.rollover(1000, 4) == pytest.approx(0.2)
    assert c.rollover(1000, 1) == pytest.approx(0.05)
    assert c.rollover(0, 24) == 0 and c.rollover(-5, 24) == 0


def test_free_model_costs_nothing():
    assert FREE.fee(1e6) == 0 and FREE.buy_price(10) == 10 and FREE.rollover(1e6, 100) == 0


def test_true_pnl_charges_fees_on_both_legs():
    # 0,01 ETH de 2000 à 2030 : gain brut 0,30 ; frais 0,26 % de 20,00 et de 20,30
    assert true_round_trip_pnl(0.01, 2000, 2030, 0.0026) == pytest.approx(0.30 - 0.0026 * 40.30)


def test_true_pnl_short_side():
    assert true_round_trip_pnl(1.0, 100, 90, 0.0, side="short") == pytest.approx(10.0)


def test_legacy_pnl_reports_profit_on_a_losing_trade():
    """Les frais retirés du PROFIT (et non du notionnel) font afficher un gain sur un trade perdant."""
    qty, entry, exit_, fee = 0.01, 2000.0, 2002.0, 0.0026       # +0,1 % brut, mais 0,52 % de frais
    assert true_round_trip_pnl(qty, entry, exit_, fee) < 0
    assert legacy_reported_pnl(qty, entry, exit_, fee, leverage=1) > 0


@pytest.mark.parametrize("leverage", [2, 3, 5])
def test_legacy_pnl_is_inflated_by_the_leverage_factor(leverage):
    qty, entry, exit_, fee = 0.01, 2000.0, 2030.0, 0.0026
    ratio = legacy_reported_pnl(qty, entry, exit_, fee, leverage) / true_round_trip_pnl(qty, entry, exit_, fee)
    assert ratio > leverage            # le levier multiplie un résultat déjà surévalué


def test_legacy_short_matches_formula():
    # bot.py ligne 549-550 : size * (1 - exit/entry) * (1 - 2*FEES) * LEVERAGE
    size_eur, entry, exit_ = 100.0, 2000.0, 1940.0
    qty = size_eur / entry
    want = size_eur * (1 - exit_ / entry) * (1 - 2 * 0.0026) * 3
    assert legacy_reported_pnl(qty, entry, exit_, 0.0026, 3, side="short") == pytest.approx(want)
