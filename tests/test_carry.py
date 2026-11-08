import numpy as np
import pandas as pd
import pytest

from tradelab import carry


def series(values, start="2023-01-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="1D", tz="UTC"))


def test_constant_funding_gives_the_expected_annual_yield():
    s = series(np.full(365, 0.0003))
    (row,) = carry.yearly_table(s, annual_friction=0.02)
    assert row.days == 365
    assert row.gross_pct == pytest.approx(10.95)
    assert row.net_pct == pytest.approx(8.95)
    assert row.negative_day_share == 0.0


def test_friction_is_subtracted_even_when_funding_is_zero():
    (row,) = carry.yearly_table(series(np.zeros(365)), 0.02)
    assert row.net_pct == pytest.approx(-2.0)


def test_years_are_reported_separately_and_partial_years_are_annualised():
    s = series(np.concatenate([np.full(365, 0.0003), np.full(91, 0.0001)]), "2023-01-01")
    rows = carry.yearly_table(s, 0.0)
    assert [r.year for r in rows] == [2023, 2024]
    assert rows[1].days == 91 and rows[1].gross_pct == pytest.approx(3.65)   # annualisé sur 91 jours


def test_negative_day_share():
    s = series([0.001, -0.001, -0.001, 0.001])
    assert carry.yearly_table(s, 0.0)[0].negative_day_share == 0.5


def test_trailing_window_only_sees_the_recent_past():
    s = series(np.concatenate([np.full(300, 0.001), np.full(65, 0.0)]))
    gross, net = carry.trailing_net_pct(s, 65, 0.02)
    assert gross == 0.0 and net == pytest.approx(-2.0)


def test_worst_window_finds_a_planted_negative_stretch():
    v = np.concatenate([np.full(200, 0.0003), np.full(100, -0.001), np.full(200, 0.0003)])
    worst, share = carry.worst_window(series(v), 90, annual_friction=0.0)
    assert worst == pytest.approx(-0.09)
    assert 0 < share < 1


def test_worst_window_needs_enough_data():
    with pytest.raises(ValueError):
        carry.worst_window(series(np.zeros(30)), 90)


def test_empty_series_is_refused():
    with pytest.raises(ValueError):
        carry.yearly_table(series([]))


def test_yield_on_capital_accounts_for_the_margin_tied_up_on_the_perpetual():
    assert carry.yield_on_capital(0.10, 1) == pytest.approx(0.05)         # spot + marge égale au notionnel
    assert carry.yield_on_capital(0.10, 2) == pytest.approx(0.10 / 1.5)
    with pytest.raises(ValueError):
        carry.yield_on_capital(0.10, 0.5)
