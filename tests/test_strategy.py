import pandas as pd
import pytest

from tradelab.strategy import (
    AccountView,
    StrategyConfig,
    StrategyState,
    adaptive_spread,
    build_grid,
    confidence_score,
    confidence_to_size_pct,
    decide,
    first_valid_index,
    get_leverage,
    order_size,
    prepare_features,
)

from conftest import make_bars, market

CFG = StrategyConfig()
H = pd.Timedelta(hours=1)


def view(equity=1000.0, free=1000.0, long_qty=0.0, short_qty=0.0):
    return AccountView(equity=equity, free_margin=free, long_qty=long_qty, short_qty=short_qty)


def built_state(m=None, v=None):
    """État après un premier cycle en RANGE à 100 : grille de 10 niveaux, taille d'ordre 4,625."""
    st = StrategyState()
    decide(m or market(), st, v or view(long_qty=10.0), CFG)
    return st


# ── fonctions pures ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("equity,expected", [(-5, 2), (0, 2), (99.99, 2), (100, 3), (499.99, 3), (500, 5), (10_000, 5)])
def test_adaptive_leverage_by_capital(equity, expected):
    assert get_leverage(equity, CFG.lev_thresholds) == expected


@pytest.mark.parametrize("atr,expected", [(0.3, 0.009), (0.8, 0.012), (1.2, 0.015), (2.0, 0.0195), (3.0, 0.024)])
def test_adaptive_spread_bands(atr, expected):
    assert adaptive_spread(atr) == pytest.approx(expected)


def test_adaptive_spread_has_a_floor():
    assert adaptive_spread(0.1, base=0.005) == pytest.approx(0.007)


def test_range_grid_is_symmetric_and_trend_up_grid_is_buy_heavy():
    rng = build_grid(100.0, 10, 0.015, "RANGE")
    up = build_grid(100.0, 10, 0.015, "TREND_UP")
    assert sum(g["type"] == "BUY" for g in rng) == 5 and sum(g["type"] == "SELL" for g in rng) == 5
    assert sum(g["type"] == "BUY" for g in up) == 7 and sum(g["type"] == "SELL" for g in up) == 3


def test_grid_levels_are_sorted_around_the_center_and_unfilled():
    g = build_grid(100.0, 10, 0.015, "RANGE")
    prices = [x["price"] for x in g]
    assert prices == sorted(prices)
    assert all(x["price"] < 100 for x in g if x["type"] == "BUY")
    assert all(x["price"] > 100 for x in g if x["type"] == "SELL")
    assert not any(x["filled"] for x in g)


def test_regime_rules():
    assert market(adx=25, close=95, ma50=100).regime(CFG) == "TREND_DOWN"
    assert market(adx=25, close=100, ma50=100).regime(CFG) == "TREND_UP"
    assert market(adx=19.9, close=50, ma50=100).regime(CFG) == "RANGE"


def test_confidence_is_bounded_and_maximal_in_the_ideal_case():
    best = market(adx=35, rsi=50, close=90, ma50=100, atr_pct=1.0)
    assert confidence_score(best, "short") == pytest.approx(1.0)
    worst = market(adx=5, rsi=80, close=101, ma50=100, atr_pct=9.0)
    assert confidence_score(worst, "short") == 0.0
    for sig in ("short", "grid_buy", "grid_sell"):
        assert 0.0 <= confidence_score(market(), sig) <= 1.0


def test_confidence_to_size_endpoints():
    assert confidence_to_size_pct(0.0) == pytest.approx(0.20) and confidence_to_size_pct(1.0) == pytest.approx(0.55)
    assert confidence_to_size_pct(0.5, 0.7, 1.3) == pytest.approx(1.0)


def test_order_size_formula_and_floor():
    assert order_size(1000, 100, 10, 5, CFG) == pytest.approx(5.0)
    assert order_size(1.0, 100_000, 10, 2, CFG) == CFG.min_order_qty


def test_features_warm_up_ends_when_the_slowest_indicator_is_ready():
    f = prepare_features(make_bars(300))
    assert first_valid_index(f) == 49                    # moyenne mobile 50 : la plus lente
    assert f[["rsi", "adx", "atr_pct", "ma50", "bbl", "bbu"]].iloc[49:].notna().all().all()


# ── grille ─────────────────────────────────────────────────────────────────────
def test_first_cycle_builds_the_grid_without_trading():
    st = StrategyState()
    orders = decide(market(), st, view(), CFG)
    assert orders == []
    assert len(st.grid) == 10 and st.regime == "RANGE" and st.center_price == 100.0
    assert st.order_size == pytest.approx(4.625)         # 1000×5/10/100 × (0,7 + 0,375×0,6)
    assert st.last_rebalance == market().ts


def test_buy_level_crossed_opens_a_long_and_arms_a_sell():
    st = built_state()
    m = market(ts=market().ts + H, close=98.0, high=99.0, low=97.9)
    orders = decide(m, st, view(long_qty=10.0), CFG)
    assert [(o.action, o.reason) for o in orders] == [("open_long", "grille achat")]
    assert orders[0].qty == pytest.approx(4.625) and orders[0].leverage == 5
    assert len(st.buy_entries) == 1 and st.buy_entries[0]["entry"] == 98.0
    assert sum(g["filled"] for g in st.grid) == 1


def test_sell_level_crossed_with_inventory_closes_a_long_and_rearms_a_buy():
    st = built_state()
    st.grid[0]["filled"] = True                           # un achat déjà pris, plus bas
    lowest_buy = min((g for g in st.grid if g["type"] == "BUY"), key=lambda g: g["price"])
    lowest_buy["filled"] = True
    m = market(ts=market().ts + H, close=101.6, high=101.7, low=100.0)
    orders = decide(m, st, view(long_qty=10.0), CFG)
    assert [(o.action, o.reason) for o in orders] == [("close_long", "grille vente")]
    assert lowest_buy["filled"] is False                  # le plus bas achat sous le prix est réarmé


def test_sell_level_without_inventory_trades_nothing_and_forces_a_rebuild():
    st = built_state()
    m = market(ts=market().ts + H, close=101.6, high=101.7, low=100.0)
    orders = decide(m, st, view(free=100.0, long_qty=0.0), CFG)
    assert orders == [] and st.last_rebalance is None and st.center_price == 0.0


def test_two_sell_levels_in_one_cycle_cannot_sell_more_than_the_inventory():
    """Régression : bot.py ne décrémente pas eth_total pendant la boucle et peut vendre à découvert."""
    st = built_state()
    size = st.order_size
    m = market(ts=market().ts + H, close=104.0, high=104.2, low=100.0)    # franchit 101,5 ET 103,0
    orders = decide(m, st, view(long_qty=size), CFG)
    sells = [o for o in orders if o.action == "close_long"]
    assert len(sells) == 1


def test_trailing_stop_sells_when_price_falls_back_from_the_peak():
    st = built_state(market(close=110.0))
    st.buy_entries = [{"entry": 100.0, "peak": 110.0}]
    m = market(ts=market().ts + H, close=108.0, high=109.0, low=107.5)
    orders = decide(m, st, view(long_qty=10.0), CFG)
    assert any(o.action == "close_long" and o.reason == "stop suiveur" for o in orders)
    assert st.buy_entries == [] or all(e["entry"] != 100.0 for e in st.buy_entries)


def test_trailing_stop_keeps_the_peak_updated():
    st = built_state()
    st.buy_entries = [{"entry": 98.0, "peak": 98.0}]
    decide(market(ts=market().ts + H, close=100.5), st, view(long_qty=10.0), CFG)
    assert st.buy_entries[0]["peak"] == 100.5


def test_inventory_below_order_size_triggers_an_auto_rebuy_and_a_rebuild():
    st = built_state()
    st.order_size = 1.0
    m = market(ts=market().ts + H)
    orders = decide(m, st, view(free=10_000.0, long_qty=0.2), CFG)
    assert orders[0].action == "open_long" and orders[0].reason == "auto-rééquilibrage"
    assert orders[0].qty == pytest.approx(3.0)
    assert st.last_rebalance is None                      # reconstruction au cycle suivant, comme bot.py


def test_grid_is_rebuilt_after_24_hours_or_a_regime_change():
    st = built_state()
    old = st.last_rebalance
    decide(market(ts=old + pd.Timedelta(hours=25)), st, view(long_qty=10.0), CFG)
    assert st.last_rebalance == old + pd.Timedelta(hours=25)
    decide(market(ts=old + pd.Timedelta(hours=26), adx=22, close=105, ma50=100), st, view(long_qty=10.0), CFG)
    assert st.regime == "TREND_UP"


# ── short ──────────────────────────────────────────────────────────────────────
DOWN = dict(adx=30.0, close=95.0, ma50=100.0, high=95.5, low=94.5, atr_pct=1.0)


def test_short_signal_opens_a_sized_short():
    st = StrategyState()
    orders = decide(market(rsi_prev=66.0, rsi=64.0, **DOWN), st, view(equity=100.0, free=100.0), CFG)
    assert len(orders) == 1 and orders[0].action == "open_short"
    assert orders[0].qty == pytest.approx(0.7237) and orders[0].leverage == 3
    assert st.in_short and st.short_entry == 95.0


def test_no_short_without_the_rsi_cross_or_a_strong_trend_or_free_margin():
    assert decide(market(rsi_prev=60.0, rsi=58.0, **DOWN), StrategyState(), view(100, 100), CFG) == []
    weak = {**DOWN, "adx": 22.0}
    assert decide(market(rsi_prev=66.0, rsi=64.0, **weak), StrategyState(), view(100, 100), CFG) == []
    assert decide(market(rsi_prev=66.0, rsi=64.0, **DOWN), StrategyState(), view(100, 9.0), CFG) == []


@pytest.mark.parametrize(
    "overrides,reason",
    [
        (dict(high=103.1, close=99.0), "stop-loss short"),     # mèche au-dessus du stop, clôture sous la MA50
        (dict(close=96.9, high=98.0), "objectif short"),
        (dict(rsi_prev=36.0, rsi=34.0, close=98.0, high=98.5), "RSI<35"),
    ],
)
def test_short_exits(overrides, reason):
    st = StrategyState(in_short=True, short_entry=100.0)
    down = {**DOWN, "close": 98.0, "high": 98.5}
    down.update(overrides)
    orders = decide(market(**down), st, view(short_qty=0.5), CFG)
    assert [(o.action, o.qty, o.reason) for o in orders] == [("close_short", 0.5, reason)]
    assert not st.in_short and st.short_entry == 0.0


def test_open_short_is_held_when_nothing_triggers():
    st = StrategyState(in_short=True, short_entry=100.0)
    down = {**DOWN, "close": 98.0, "high": 98.5}
    assert decide(market(rsi=45.0, rsi_prev=46.0, **down), st, view(short_qty=0.5), CFG) == []
    assert st.in_short


def test_short_is_covered_when_the_regime_turns_back():
    st = StrategyState(in_short=True, short_entry=100.0)
    orders = decide(market(adx=15.0), st, view(short_qty=0.5), CFG)
    assert orders[0].action == "close_short" and "régime" in orders[0].reason


def test_trend_down_suspends_the_grid():
    st = built_state()
    assert st.grid
    decide(market(ts=market().ts + H, **DOWN), st, view(long_qty=10.0), CFG)
    assert st.grid == [] and st.last_rebalance is None


def test_state_resyncs_when_the_account_has_no_short():
    st = StrategyState(in_short=True, short_entry=100.0)
    decide(market(**DOWN, rsi=50.0, rsi_prev=50.0), st, view(short_qty=0.0), CFG)
    assert not st.in_short


# ── refus d'ordre ──────────────────────────────────────────────────────────────
def test_rejected_short_restores_the_state():
    st = StrategyState()
    (order,) = decide(market(rsi_prev=66.0, rsi=64.0, **DOWN), st, view(100, 100), CFG)
    st.revert(order)
    assert not st.in_short and st.short_entry == 0.0


def test_rejected_grid_buy_rearms_the_level_and_forgets_the_entry():
    st = built_state()
    (order,) = decide(market(ts=market().ts + H, close=98.0, high=99.0, low=97.9), st, view(long_qty=10.0), CFG)
    st.revert(order)
    assert not any(g["filled"] and g["type"] == "BUY" for g in st.grid)
    assert st.buy_entries == []


def test_liquidation_reset_clears_everything_that_depended_on_positions():
    st = built_state()
    st.in_short, st.short_entry, st.buy_entries = True, 100.0, [{"entry": 1, "peak": 1}]
    st.reset_after_liquidation()
    assert (st.in_short, st.grid, st.buy_entries, st.last_rebalance) == (False, [], [], None)


def test_fixed_leverage_overrides_the_capital_tiers():
    st = StrategyState()
    decide(market(), st, view(equity=1000.0), StrategyConfig(fixed_leverage=1))
    assert st.order_size == pytest.approx(1.0 * 0.925)    # 1000×1/10/100 × 0,925
