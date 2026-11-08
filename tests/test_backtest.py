import numpy as np
import pandas as pd
import pytest

from tradelab.backtest import run_backtest
from tradelab.costs import FREE, CostModel
from tradelab.strategy import StrategyConfig

from conftest import make_bars


@pytest.fixture(scope="module")
def walk():
    return make_bars(4000, seed=3, vol=0.006)


@pytest.mark.parametrize("cfg", [StrategyConfig(fixed_leverage=1), StrategyConfig()], ids=["x1", "adaptatif"])
def test_accounting_identity_without_costs(walk, cfg):
    """Sans coût, l'équité vaut exactement le produit des (1 + exposition × rendement de bougie)."""
    r = run_backtest(walk, cfg, FREE, 50.0)
    expected = 50.0 * np.cumprod(1.0 + r.exposure.to_numpy() * r.bar_return.to_numpy())
    assert np.allclose(r.equity.to_numpy(), expected, rtol=1e-9, atol=1e-9)
    assert r.liquidations == 0


def test_runs_are_deterministic(walk):
    a = run_backtest(walk, StrategyConfig(), CostModel(), 50.0)
    b = run_backtest(walk, StrategyConfig(), CostModel(), 50.0)
    pd.testing.assert_series_equal(a.equity, b.equity)
    assert a.fills == b.fills


def test_no_lookahead_changing_the_future_does_not_change_the_past(walk):
    cut = 2500
    altered = walk.copy()
    altered.iloc[cut:, :4] = altered.iloc[cut:, :4] * 1.35
    a = run_backtest(walk, StrategyConfig(), CostModel(), 50.0)
    b = run_backtest(altered, StrategyConfig(), CostModel(), 50.0)
    start = len(walk) - len(a.equity)
    n = cut - start - 2
    assert n > 500
    pd.testing.assert_series_equal(a.equity.iloc[:n], b.equity.iloc[:n])


def test_orders_execute_at_the_next_open_not_at_the_decision_close(walk):
    r = run_backtest(walk, StrategyConfig(), FREE, 50.0)
    regular = [f for f in r.fills if f.action != "liquidation" and f.reason != "fin du backtest"]
    assert len(regular) > 20
    for f in regular:
        assert f.price == pytest.approx(walk.loc[f.ts, "open"])


def test_costs_reduce_the_result(walk):
    free = run_backtest(walk, StrategyConfig(fixed_leverage=1), FREE, 50.0)
    paid = run_backtest(walk, StrategyConfig(fixed_leverage=1), CostModel(), 50.0)
    assert paid.equity.iloc[-1] < free.equity.iloc[-1]
    assert paid.fees_paid > 0 and free.fees_paid == 0


def test_rollover_only_exists_with_leverage(walk):
    levered = run_backtest(walk, StrategyConfig(fixed_leverage=2), CostModel(), 50.0)
    unlevered = run_backtest(walk, StrategyConfig(fixed_leverage=1), CostModel(), 50.0)
    assert levered.rollover_paid > 0 and unlevered.rollover_paid == 0


def test_start_date_means_a_fresh_account_and_no_earlier_trades(walk):
    start = walk.index[2000]
    r = run_backtest(walk, StrategyConfig(), CostModel(), 50.0, start=start)
    assert r.equity.index[0] >= start
    assert all(f.ts >= start for f in r.fills)
    assert abs(r.equity.iloc[0] - 50.0) / 50.0 < 0.05           # compte neuf : peu de mouvement sur la 1re bougie


def test_start_date_accepts_a_string(walk):
    r = run_backtest(walk, StrategyConfig(), FREE, 50.0, start="2022-03-01")
    assert r.equity.index[0] >= pd.Timestamp("2022-03-01", tz="UTC")


def test_too_short_history_is_refused():
    with pytest.raises(ValueError):
        run_backtest(make_bars(40), StrategyConfig(), FREE, 50.0)


def test_summary_properties(walk):
    r = run_backtest(walk, StrategyConfig(), CostModel(), 50.0)
    assert r.trades == sum(f.action == "close" for f in r.fills)
    assert r.fees_paid == pytest.approx(sum(f.fee for f in r.fills))
    assert r.total_return == pytest.approx(r.equity.iloc[-1] / 50.0 - 1)


def _flat_then_crash(low: float, close: float, n: int = 260, k: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2022-01-01", periods=n, freq="1h", tz="UTC")
    price = np.full(n, 100.0)
    df = pd.DataFrame({"open": price, "high": price, "low": price, "close": price, "volume": 1.0}, index=idx)
    df.iloc[k, df.columns.get_loc("low")] = low             # mèche, puis le prix reste bas
    df.iloc[k, df.columns.get_loc("close")] = close
    df.iloc[k + 1 :, :4] = close
    return df


def test_a_crash_liquidates_the_account_and_the_run_continues():
    # Série plate : le bot achète 3 × la taille de grille (auto-rééquilibrage), puis le prix chute de 90 %.
    df = _flat_then_crash(low=10.0, close=15.0)
    r = run_backtest(df, StrategyConfig(fixed_leverage=3), CostModel(), 1000.0)
    assert r.liquidations == 1
    assert r.equity.iloc[-1] < 1000.0
    assert r.equity.index[-1] == df.index[-1]


def test_ruined_account_stops_trading():
    df = _flat_then_crash(low=10.0, close=15.0)
    r = run_backtest(df, StrategyConfig(fixed_leverage=3), CostModel(liquidation_fee=0.9), 1000.0)
    assert r.ruined and r.equity.iloc[-1] == 0.0


def test_a_mild_dip_does_not_liquidate():
    df = _flat_then_crash(low=80.0, close=85.0)
    r = run_backtest(df, StrategyConfig(fixed_leverage=3), CostModel(), 1000.0)
    assert r.liquidations == 0
