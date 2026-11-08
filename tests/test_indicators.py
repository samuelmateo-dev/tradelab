import numpy as np
import pandas as pd
import pytest

from tradelab.indicators import adx, atr, bollinger, rsi, sma, true_range

from conftest import make_bars


def naive_rsi(close, n):
    """RSI de Wilder écrit à la main : moyenne simple d'amorçage puis récurrence."""
    d = np.diff(close)
    up, dn = np.maximum(d, 0), np.maximum(-d, 0)
    out = [np.nan] * len(close)
    # même convention que pandas ewm(adjust=False, min_periods=n) : amorçage par la récurrence depuis le 1er écart
    au, ad = up[0], dn[0]
    for i in range(1, len(d)):
        au = au + (up[i] - au) / n
        ad = ad + (dn[i] - ad) / n
        if i + 1 >= n:
            out[i + 1] = 100 * au / (au + ad) if au + ad > 0 else 50.0
    return np.array(out)


def naive_atr(h, l, c, n):
    tr = [h[0] - l[0]] + [max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])) for i in range(1, len(c))]
    out = [np.nan] * len(c)
    a = sum(tr[:n]) / n
    out[n - 1] = a
    for i in range(n, len(c)):
        a = (a * (n - 1) + tr[i]) / n
        out[i] = a
    return np.array(out)


def naive_adx(h, l, c, n):
    """ADX de Wilder par définition, avec des listes Python (structure volontairement différente)."""
    size = len(c)
    tr, pdm, mdm = [0.0], [0.0], [0.0]
    for i in range(1, size):
        tr.append(max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])))
        up, dn = h[i] - h[i - 1], l[i - 1] - l[i]
        pdm.append(up if up > dn and up > 0 else 0.0)
        mdm.append(dn if dn > up and dn > 0 else 0.0)
    dxs = {}
    s_tr, s_p, s_m = sum(tr[1 : n + 1]), sum(pdm[1 : n + 1]), sum(mdm[1 : n + 1])
    for i in range(n, size):
        if i > n:
            s_tr += tr[i] - s_tr / n
            s_p += pdm[i] - s_p / n
            s_m += mdm[i] - s_m / n
        dip, dim = 100 * s_p / s_tr, 100 * s_m / s_tr
        dxs[i] = 100 * abs(dip - dim) / (dip + dim) if dip + dim else 0.0
    out = [np.nan] * size
    cur = sum(dxs[i] for i in range(n, 2 * n)) / n
    out[2 * n - 1] = cur
    for i in range(2 * n, size):
        cur = (cur * (n - 1) + dxs[i]) / n
        out[i] = cur
    return np.array(out)


@pytest.fixture(scope="module")
def ohlc():
    return make_bars(600, seed=7, vol=0.01)


def test_rsi_matches_hand_written_wilder(ohlc):
    got = rsi(ohlc["close"], 14).to_numpy()
    want = naive_rsi(ohlc["close"].to_numpy(), 14)
    assert np.allclose(got[14:], want[14:], atol=1e-9)


def test_rsi_is_nan_until_window_is_filled(ohlc):
    r = rsi(ohlc["close"], 21)
    assert r.iloc[:20].isna().all() and r.iloc[21:].notna().all()


def test_rsi_stays_between_0_and_100(ohlc):
    r = rsi(ohlc["close"], 14).dropna()
    assert ((r >= 0) & (r <= 100)).all()


def test_rsi_extremes():
    up = pd.Series(np.arange(1.0, 80.0))
    flat = pd.Series([5.0] * 80)
    assert rsi(up, 14).iloc[-1] == pytest.approx(100.0)
    assert rsi(flat, 14).iloc[-1] == 50.0          # aucune variation : neutre, pas de division par zéro


def test_atr_matches_hand_written(ohlc):
    got = atr(ohlc["high"], ohlc["low"], ohlc["close"], 14).to_numpy()
    want = naive_atr(ohlc["high"].to_numpy(), ohlc["low"].to_numpy(), ohlc["close"].to_numpy(), 14)
    assert np.allclose(got[13:], want[13:], atol=1e-9)


def test_atr_constant_range():
    n = 60
    c = pd.Series([100.0] * n)
    a = atr(c + 1.0, c - 1.0, c, 14)
    assert a.dropna().eq(2.0).all()


def test_true_range_uses_gap_from_previous_close():
    h = pd.Series([10.0, 15.0]); l = pd.Series([9.0, 14.0]); c = pd.Series([9.5, 14.5])
    assert true_range(h, l, c).iloc[1] == pytest.approx(15.0 - 9.5)   # écart avec la clôture précédente


@pytest.mark.parametrize("seed", [7, 8, 9, 10, 11, 12, 13, 14])
def test_adx_matches_hand_written(seed):
    """Plusieurs séries : sur une seule, un décalage d'amorçage peut passer inaperçu si la bougie d'amorçage est neutre."""
    df = make_bars(400, seed=seed, vol=0.01)
    got = adx(df["high"], df["low"], df["close"], 14).to_numpy()
    want = naive_adx(df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy(), 14)
    assert np.isnan(got[:26]).all()
    assert np.allclose(got[27:], want[27:], rtol=1e-9, atol=1e-9)


def test_adx_high_in_trend_and_low_in_range():
    n = 200
    trend = pd.Series(100.0 + np.arange(n) * 1.0)
    t_adx = adx(trend + 0.5, trend - 0.5, trend, 14).iloc[-1]
    saw = pd.Series(100.0 + np.where(np.arange(n) % 2 == 0, 1.0, -1.0))
    r_adx = adx(saw + 0.5, saw - 0.5, saw, 14).iloc[-1]
    assert t_adx > 60 and r_adx < 10


def test_adx_too_short_series_is_all_nan():
    s = pd.Series(np.linspace(1, 2, 20))
    assert adx(s + 0.1, s - 0.1, s, 14).isna().all()


def test_bollinger_collapses_on_constant_series_and_uses_population_std():
    flat = pd.Series([50.0] * 30)
    lo, hi = bollinger(flat, 14, 2.5)
    assert (hi - lo).dropna().eq(0).all()
    x = pd.Series(np.arange(20.0))
    lo, hi = bollinger(x, 14, 2.0)
    w = x.iloc[-14:]
    assert hi.iloc[-1] == pytest.approx(w.mean() + 2 * w.std(ddof=0))


@pytest.mark.parametrize("fn", ["rsi", "atr", "adx", "sma", "bollinger"])
def test_indicators_are_causal(ohlc, fn):
    """La valeur à t ne change pas si l'on modifie les bougies futures : pas de regard vers l'avenir."""
    def run(df):
        c, h, l = df["close"], df["high"], df["low"]
        return {"rsi": lambda: rsi(c, 14), "atr": lambda: atr(h, l, c, 14), "adx": lambda: adx(h, l, c, 14),
                "sma": lambda: sma(c, 20), "bollinger": lambda: bollinger(c, 14, 2.5)[0]}[fn]()
    full = run(ohlc)
    cut = 400
    altered = ohlc.copy()
    altered.iloc[cut:] = altered.iloc[cut:] * 1.7
    other = run(altered)
    assert np.allclose(full.iloc[:cut].dropna(), other.iloc[:cut].dropna(), equal_nan=True)


def test_window_of_100_bars_stays_close_to_full_history(ohlc):
    """Le bot réel calcule ses indicateurs sur 100 bougies ; nous, sur tout l'historique. L'écart est faible."""
    c, h, l = ohlc["close"], ohlc["high"], ohlc["low"]
    full_rsi, full_adx = rsi(c, 21), adx(h, l, c, 14)
    worst_rsi = worst_adx = 0.0
    for end in range(300, 600, 25):
        w = slice(end - 100, end)
        worst_rsi = max(worst_rsi, abs(rsi(c.iloc[w], 21).iloc[-1] - full_rsi.iloc[end - 1]))
        worst_adx = max(worst_adx, abs(adx(h.iloc[w], l.iloc[w], c.iloc[w], 14).iloc[-1] - full_adx.iloc[end - 1]))
    assert worst_rsi < 3.0 and worst_adx < 6.0
