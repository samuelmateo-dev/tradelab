"""Indicateurs techniques (lissage de Wilder), sans dépendance externe.

Tous sont CAUSALS : la valeur à la bougie *t* n'utilise que les bougies ≤ *t*. C'est ce qui permet
de les précalculer sur tout l'historique sans regarder le futur. Les valeurs restent ``NaN`` tant
que la fenêtre d'amorçage n'est pas remplie.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """RSI de Wilder : moyennes mobiles exponentielles de coefficient 1/window."""
    diff = close.diff()
    up = diff.clip(lower=0.0)
    down = (-diff).clip(lower=0.0)
    avg_up = up.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_down = down.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    total = avg_up + avg_down
    # Aucun mouvement sur toute la fenêtre : RSI neutre plutôt qu'une division par zéro.
    return (100.0 * avg_up / total).where(total > 0, 50.0).where(avg_up.notna())


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    tr.iloc[0] = high.iloc[0] - low.iloc[0]
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range : moyenne simple des *window* premiers TR, puis lissage de Wilder."""
    tr = true_range(high, low, close).to_numpy(dtype=float)
    out = np.full(len(tr), np.nan)
    if len(tr) >= window:
        out[window - 1] = tr[:window].mean()
        for i in range(window, len(tr)):
            out[i] = (out[i - 1] * (window - 1) + tr[i]) / window
    return pd.Series(out, index=close.index)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average Directional Index de Wilder (première valeur à l'indice 2·window − 2)."""
    h = high.to_numpy(dtype=float)
    l = low.to_numpy(dtype=float)
    n = len(h)
    out = np.full(n, np.nan)
    if n < 2 * window:
        return pd.Series(out, index=close.index)

    tr = true_range(high, low, close).to_numpy(dtype=float)
    up_move = np.zeros(n)
    down_move = np.zeros(n)
    up_move[1:] = h[1:] - h[:-1]
    down_move[1:] = l[:-1] - l[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    # Lissage de Wilder en somme : amorçage sur les bougies 1..window (la bougie 0 n'a pas de précédent).
    s_tr = tr[1 : window + 1].sum()
    s_p = plus_dm[1 : window + 1].sum()
    s_m = minus_dm[1 : window + 1].sum()
    dx = np.full(n, np.nan)
    for i in range(window, n):
        if i > window:
            s_tr = s_tr - s_tr / window + tr[i]
            s_p = s_p - s_p / window + plus_dm[i]
            s_m = s_m - s_m / window + minus_dm[i]
        if s_tr > 0:
            di_p = 100.0 * s_p / s_tr
            di_m = 100.0 * s_m / s_tr
            denom = di_p + di_m
            dx[i] = 100.0 * abs(di_p - di_m) / denom if denom > 0 else 0.0
        else:
            dx[i] = 0.0

    first = 2 * window - 1
    out[first] = dx[window : first + 1].mean()
    for i in range(first + 1, n):
        out[i] = (out[i - 1] * (window - 1) + dx[i]) / window
    return pd.Series(out, index=close.index)


def bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0) -> tuple[pd.Series, pd.Series]:
    """Bandes de Bollinger (écart-type de population, comme la bibliothèque ``ta``)."""
    mid = close.rolling(window).mean()
    std = close.rolling(window).std(ddof=0)
    return mid - num_std * std, mid + num_std * std
