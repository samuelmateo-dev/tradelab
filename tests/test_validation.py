import numpy as np
import pandas as pd
import pytest

from tradelab.backtest import BacktestResult
from tradelab.validation import (
    bootstrap_mean,
    circular_shift_placebo,
    judge,
    lower_bound,
    monthly_returns,
    window_stats,
)

N = 24 * 365 * 3 + 24 * 30          # ≈ 37 mois de bougies horaires


def fake_result(exposure, bar_return, start="2020-01-01", initial=100.0) -> BacktestResult:
    idx = pd.date_range(start, periods=len(exposure), freq="1h", tz="UTC")
    eq = initial * np.cumprod(1.0 + exposure * bar_return)
    return BacktestResult(
        equity=pd.Series(eq, idx), exposure=pd.Series(exposure, idx), gross_exposure=pd.Series(np.abs(exposure), idx),
        bar_return=pd.Series(bar_return, idx), initial_equity=initial, fills=[], rejected=0, liquidations=0,
        rollover_paid=0.0, ruined=False,
    )


def prices(bar_return, start="2020-01-01"):
    idx = pd.date_range(start, periods=len(bar_return), freq="1h", tz="UTC")
    return pd.Series(1000.0 * np.cumprod(1.0 + bar_return), idx)


@pytest.fixture(scope="module")
def returns():
    return np.random.default_rng(11).normal(0.0, 0.004, N)


# ── briques ────────────────────────────────────────────────────────────────────
def test_monthly_returns_known_values():
    idx = pd.date_range("2024-01-01", "2024-03-31", freq="1D", tz="UTC")
    s = pd.Series(100.0, idx)
    s[s.index >= "2024-01-31"] = 200.0                    # ×2 en janvier
    s[s.index >= "2024-03-01"] = 100.0                    # −50 % en mars
    m = monthly_returns(s, base=100.0)
    assert m.round(6).tolist() == [1.0, 0.0, -0.5]


def test_bootstrap_is_seeded_and_centered_on_the_sample_mean():
    x = np.random.default_rng(0).normal(0.01, 0.05, 60)
    a, b = bootstrap_mean(x, 3000, seed=5), bootstrap_mean(x, 3000, seed=5)
    assert np.array_equal(a, b) and not np.array_equal(a, bootstrap_mean(x, 3000, seed=6))
    assert abs(a.mean() - x.mean()) < 0.003


def test_block_bootstrap_is_wider_for_autocorrelated_series():
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.normal(0, 0.01, 120))               # fortement autocorrélé
    assert bootstrap_mean(x, 4000, 2, block=6).std() > bootstrap_mean(x, 4000, 2, block=1).std()


def test_bootstrap_rejects_empty_sample():
    with pytest.raises(ValueError):
        bootstrap_mean(np.array([]))


def test_lower_bound_of_a_constant_positive_sample_is_that_constant():
    assert lower_bound(np.full(30, 0.02), 0.05) == pytest.approx(0.02)


def test_lower_bound_of_zero_mean_noise_is_negative():
    x = np.random.default_rng(3).normal(0, 0.05, 40)
    assert lower_bound(x, 0.025) < 0


def test_placebo_detects_real_timing_skill(returns):
    exposure = (returns > 0).astype(float)                 # clairvoyant
    assert circular_shift_placebo(exposure, returns, n=500)["p_value"] < 0.01


def test_placebo_does_not_reward_unrelated_exposure(returns):
    exposure = np.random.default_rng(4).uniform(0, 1, len(returns))
    assert circular_shift_placebo(exposure, returns, n=500)["p_value"] > 0.05


def test_placebo_p_value_is_never_zero(returns):
    p = circular_shift_placebo((returns > 0).astype(float), returns, n=200)["p_value"]
    assert p >= 1 / 201


def test_placebo_input_checks(returns):
    with pytest.raises(ValueError):
        circular_shift_placebo(np.ones(10), np.ones(11))
    with pytest.raises(ValueError):
        circular_shift_placebo(np.ones(500), np.ones(500))


# ── verdict ────────────────────────────────────────────────────────────────────
def both(e, r):
    """DEV en 2016-2019 et VAL en 2023-2026 : un seul jeu de cours couvre les deux fenêtres, comme dans la CLI."""
    dev, val = fake_result(e, r, "2016-01-01"), fake_result(e, r, "2023-02-01")
    close = pd.concat([prices(r, "2016-01-01"), prices(r, "2023-02-01")])
    return dev, val, close


def window(e, r, start):
    return fake_result(e, r, start), prices(r, start)


def test_a_clairvoyant_strategy_is_demonstrated(returns):
    e = (returns > 0).astype(float)
    dev, val, close = both(e, returns)
    v = judge("clairvoyant", dev, val, close, n_boot=1000)
    assert v.absolute_ok and v.excess_ok and v.placebo_ok and v.demonstrated


def test_unrelated_exposure_is_not_demonstrated(returns):
    e = np.random.default_rng(8).uniform(0, 1, len(returns))
    dev, val, close = both(e, returns)
    v = judge("bruit", dev, val, close, n_boot=1000)
    assert not v.demonstrated


def test_anti_clairvoyant_strategy_loses_everywhere(returns):
    e = (returns < 0).astype(float)
    dev, val, close = both(e, returns)
    v = judge("inverse", dev, val, close, n_boot=1000)
    assert not v.absolute_ok and not v.placebo_ok and not v.demonstrated


def test_too_short_a_validation_window_cannot_demonstrate_anything(returns):
    e = (returns > 0).astype(float)
    short = slice(0, 24 * 30 * 12)                         # 12 mois de validation seulement
    dev = fake_result(e, returns, "2016-01-01")
    val = fake_result(e[short], returns[short], "2023-02-01")
    close = pd.concat([prices(returns, "2016-01-01"), prices(returns[short], "2023-02-01")])
    v = judge("trop court", dev, val, close, n_boot=500)
    assert not v.val.sufficient and not v.demonstrated


def test_bonferroni_divides_alpha_by_the_number_of_hypotheses(returns):
    e = np.random.default_rng(8).uniform(0, 1, len(returns))
    dev, val, close = both(e, returns)
    assert judge("h", dev, val, close, n_hypotheses=4, n_boot=200).alpha_adjusted == pytest.approx(0.0125)


def test_window_stats_reports_buy_and_hold_and_exposure(returns):
    e = np.full(len(returns), 0.5)
    res, close = window(e, returns, "2020-01-01")
    w = window_stats("X", res, close, 0.05, n_boot=300)
    assert w.mean_exposure == pytest.approx(0.5)
    assert w.hold_return == pytest.approx(close.iloc[-1] / close.iloc[0] - 1)
    assert w.months >= 36 and w.sufficient
    # exposition constante de 0,5 : l'excès sur la détention de même exposition est (presque) nul
    assert abs(w.mean_excess) < 0.01
