"""Données FICTIVES, reproductibles, pour essayer l'outil sans télécharger quoi que ce soit.

Ce ne sont pas des prix réels et elles ne disent rien d'un vrai marché : une marche aléatoire à
volatilité variable, avec des queues épaisses, et un financement quotidien fictif. Elles servent
à faire tourner la ligne de commande et les tests de bout en bout. Pour juger une stratégie, il
faut des données réelles (voir le README).

Tout est déterminé par la graine : mêmes paramètres, mêmes fichiers, octet pour octet.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

START = "2020-01-01 00:00"
END = "2026-10-09 23:00"
BARS_SEED = 2020
FUNDING_SEED = 7
BARS_FILE = "eth_1h.csv.gz"
FUNDING_FILE = "btc_funding.csv"


def synthetic_bars(
    start: str = START,
    end: str = END,
    seed: int = BARS_SEED,
    start_price: float = 130.0,
    annual_vol: float = 0.75,
    annual_drift: float = 0.71,
) -> pd.DataFrame:
    """Bougies d'une heure (UTC) sans trou : open, high, low, close, volume.

    Le rendement de chaque heure est un tirage de Student (4 degrés de liberté, donc des queues
    épaisses) multiplié par une volatilité qui dérive lentement. L'ouverture est la clôture
    précédente. Mèches et volume sont tirés au hasard autour du corps de la bougie.
    """
    index = pd.date_range(start, end, freq="1h", tz="UTC", name="time")
    n = len(index)
    rng = np.random.default_rng(seed)

    # Régime de volatilité : processus AR(1) sur le logarithme, très persistant (à l'échelle de l'heure).
    phi = 0.9995
    shocks = rng.standard_normal(n) * 0.01
    log_vol = np.empty(n)
    log_vol[0] = 0.0
    for i in range(1, n):
        log_vol[i] = phi * log_vol[i - 1] + shocks[i]
    vol = np.exp(log_vol)
    vol *= (annual_vol / np.sqrt(8760)) / np.sqrt(np.mean(vol**2))  # écart-type annualisé = annual_vol

    t = rng.standard_t(4, size=n) / np.sqrt(2.0)  # variance de Student(4) = 2
    ret = annual_drift / 8760 - 0.5 * vol**2 + vol * t

    close = start_price * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[start_price], close[:-1]])
    body_hi = np.maximum(open_, close)
    body_lo = np.minimum(open_, close)
    wick_hi = np.abs(rng.standard_normal(n)) * vol * 0.6
    wick_lo = np.abs(rng.standard_normal(n)) * vol * 0.6
    high = body_hi * (1 + wick_hi)
    low = body_lo * (1 - wick_lo)
    volume = np.exp(rng.normal(8.0, 0.6, n)) * (1 + 40 * np.abs(ret) / vol.mean())

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=index
    )
    # L'arrondi est monotone : il ne peut pas rendre une bougie incohérente.
    return df.round({"open": 2, "high": 2, "low": 2, "close": 2, "volume": 2})


def synthetic_funding(
    start: str = "2020-01-01",
    end: str = "2026-09-30",
    seed: int = FUNDING_SEED,
) -> pd.Series:
    """Somme quotidienne fictive du financement d'un perpétuel (fraction du notionnel par jour).

    Un niveau moyen qui varie lentement (parfois négatif), plus un bruit quotidien : de quoi
    exercer ``tradelab carry``, pas de quoi estimer un rendement.
    """
    index = pd.date_range(start, end, freq="1D", tz="UTC", name="date")
    n = len(index)
    rng = np.random.default_rng(seed)
    level = np.empty(n)
    level[0] = 0.00015
    for i in range(1, n):
        level[i] = level[i - 1] + 0.02 * (0.00012 - level[i - 1]) + rng.normal(0, 0.000025)
    daily = level + rng.normal(0, 0.00012, n)
    return pd.Series(daily, index=index, name="funding_sum").round(8)


def write_files(out: str | Path) -> list[Path]:
    """Écrit les deux fichiers fictifs dans ``out`` (créé au besoin). Sortie identique à chaque appel."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    bars_path, funding_path = out / BARS_FILE, out / FUNDING_FILE
    synthetic_bars().to_csv(
        bars_path, compression={"method": "gzip", "mtime": 0}, float_format="%.8g"
    )
    synthetic_funding().to_csv(funding_path, float_format="%.8g")
    return [bars_path, funding_path]
