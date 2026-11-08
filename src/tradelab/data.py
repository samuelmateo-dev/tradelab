"""Chargement, rééchantillonnage et contrôle qualité des bougies.

Format d'origine (.npz) : ``ts`` en MINUTES depuis 1970 (et non en millisecondes), puis
``o h l c v`` et ``tb`` (volume acheteur au prix du marché). Le contrôle qualité existe parce
que les trous et les bougies incohérentes faussent un backtest sans que rien ne plante.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

OHLCV = ["open", "high", "low", "close", "volume"]
_NPZ_COLUMNS = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "tb": "taker_buy"}


def load_klines_npz(path: str | Path) -> pd.DataFrame:
    """Lit un fichier mensuel de bougies d'une minute. Refuse les objets « pickle » (code arbitraire)."""
    with np.load(path, allow_pickle=False) as z:
        if "ts" not in z.files:
            raise ValueError(f"{path} : clé « ts » absente (fichier de bougies attendu).")
        cols = {name: z[key] for key, name in _NPZ_COLUMNS.items() if key in z.files}
        index = pd.to_datetime(z["ts"].astype("int64") * 60, unit="s", utc=True)
    return pd.DataFrame(cols, index=index).rename_axis("time")


def load_month_dir(directory: str | Path) -> pd.DataFrame:
    """Concatène tous les ``*.npz`` d'un dossier, triés, sans doublon d'horodatage (le dernier gagne)."""
    files = sorted(Path(directory).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"Aucun fichier .npz dans {directory}")
    df = pd.concat([load_klines_npz(f) for f in files])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def resample_ohlcv(df: pd.DataFrame, rule: str = "1h") -> pd.DataFrame:
    """Agrège en bougies plus longues. Une période sans aucune bougie source est supprimée, jamais inventée."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    out = df[list(agg)].resample(rule, label="left", closed="left").agg(agg)
    return out.dropna(subset=["open", "high", "low", "close"])


def load_csv(path: str | Path) -> pd.DataFrame:
    """Lit un fichier d'exemple (``time`` ou ``date`` en première colonne, UTC)."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.sort_index()


@dataclass(frozen=True)
class QualityReport:
    rows: int
    start: pd.Timestamp
    end: pd.Timestamp
    expected_rows: int
    missing_rows: int
    duplicated_index: int
    non_positive_prices: int
    inconsistent_ohlc: int

    @property
    def missing_share(self) -> float:
        return self.missing_rows / self.expected_rows if self.expected_rows else 0.0

    @property
    def clean(self) -> bool:
        return self.duplicated_index == 0 and self.non_positive_prices == 0 and self.inconsistent_ohlc == 0


def quality_report(df: pd.DataFrame, freq: str = "1h") -> QualityReport:
    """Compte les anomalies sans rien corriger : on veut savoir, pas masquer."""
    if df.empty:
        raise ValueError("Jeu de données vide.")
    step = pd.tseries.frequencies.to_offset(freq)
    expected = len(pd.date_range(df.index[0], df.index[-1], freq=step))
    prices = df[["open", "high", "low", "close"]]
    bad_hl = (df["high"] < df[["open", "close"]].max(axis=1) - 1e-12) | (
        df["low"] > df[["open", "close"]].min(axis=1) + 1e-12
    )
    return QualityReport(
        rows=len(df),
        start=df.index[0],
        end=df.index[-1],
        expected_rows=expected,
        missing_rows=max(expected - len(df), 0),
        duplicated_index=int(df.index.duplicated().sum()),
        non_positive_prices=int((prices <= 0).any(axis=1).sum()),
        inconsistent_ohlc=int(bad_hl.sum()),
    )
