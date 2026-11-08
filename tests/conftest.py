import numpy as np
import pandas as pd
import pytest

from tradelab.strategy import Market


def make_bars(n: int = 3000, seed: int = 1, start: str = "2022-01-01", vol: float = 0.004, drift: float = 0.0) -> pd.DataFrame:
    """Marche aléatoire horaire SANS trou, où l'ouverture d'une bougie égale la clôture de la précédente."""
    rng = np.random.default_rng(seed)
    close = 1000.0 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    open_ = np.concatenate([[1000.0], close[:-1]])
    wiggle = np.abs(rng.normal(0, vol / 2, n))
    high = np.maximum(open_, close) * (1 + wiggle)
    low = np.minimum(open_, close) * (1 - wiggle)
    idx = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": 1.0}, index=idx)


def market(**kw) -> Market:
    """Marché neutre (RANGE, au milieu des bandes) dont on surcharge les champs utiles."""
    base = dict(ts=pd.Timestamp("2024-01-01", tz="UTC"), close=100.0, high=100.5, low=99.5, adx=15.0,
                atr_pct=1.2, rsi=50.0, rsi_prev=50.0, ma50=100.0, bbl=97.0, bbu=103.0)
    base.update(kw)
    return Market(**base)


@pytest.fixture
def bars() -> pd.DataFrame:
    return make_bars()
