"""Juge statistique d'une stratégie : peut-on distinguer son résultat du hasard et de la simple détention ?

Protocole (inspiré des « protocoles figés » du projet d'origine) :
1. la stratégie est GELÉE avant le test (aucun réglage n'est appris ici) ;
2. la période est coupée en DEV (contexte) et VAL (juge) ; VAL démarre avec un capital neuf, pour que
   le verdict ne dépende pas d'une ruine survenue avant ;
3. l'incertitude est mesurée sur les MOIS (blocs entiers), pas sur les heures, qui ne sont pas indépendantes ;
4. trois conditions doivent TOUTES tenir pour déclarer « démontré » :
   a) le rendement mensuel moyen net a une borne basse > 0 ;
   b) l'excès sur la détention de même exposition moyenne a une borne basse > 0 ;
   c) le test de placebo (même profil d'exposition, rythme décalé au hasard) donne p < α ;
5. α est corrigé de Bonferroni pour le nombre d'hypothèses testées.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import BacktestResult

MIN_MONTHS = 24


# ── briques statistiques ───────────────────────────────────────────────────────
def monthly_returns(values: pd.Series, base: float) -> pd.Series:
    """Rendements mensuels d'une série de valeurs. ``base`` = valeur avant le premier mois."""
    ends = values.resample("MS").last().dropna()
    prev = ends.shift(1)
    prev.iloc[0] = base
    return (ends / prev - 1.0).rename("monthly_return")


def bootstrap_mean(x: np.ndarray, n_boot: int = 5000, seed: int = 42, block: int = 1) -> np.ndarray:
    """Distribution bootstrap de la moyenne ; ``block`` > 1 tire des suites de mois consécutifs."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        raise ValueError("Échantillon vide.")
    rng = np.random.default_rng(seed)
    block = max(1, min(block, n))
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    return x[idx].mean(axis=1)


def lower_bound(x: np.ndarray, alpha: float, n_boot: int = 5000, seed: int = 42, block: int = 1) -> float:
    """Borne basse unilatérale au niveau ``alpha`` de la moyenne."""
    return float(np.quantile(bootstrap_mean(x, n_boot, seed, block), alpha))


def circular_shift_placebo(
    exposure: np.ndarray, bar_return: np.ndarray, n: int = 2000, seed: int = 42, min_shift: int = 24 * 30
) -> dict[str, float]:
    """Placebo : rejoue le MÊME profil d'exposition avec un rythme décalé au hasard.

    Mesure brute (avant frais) Σ exposition × rendement de bougie. Si le timing de la stratégie ne
    valait rien, un décalage circulaire aléatoire ferait aussi bien. ``p`` = part des placebos qui
    font au moins aussi bien que la stratégie (avec correction +1 pour ne jamais annoncer p = 0).
    """
    e = np.asarray(exposure, dtype=float)
    r = np.asarray(bar_return, dtype=float)
    if len(e) != len(r):
        raise ValueError("exposition et rendements de longueurs différentes.")
    if len(e) <= 2 * min_shift:
        raise ValueError("Série trop courte pour un décalage circulaire significatif.")
    actual = float(np.dot(e, r))
    rng = np.random.default_rng(seed)
    shifts = rng.integers(min_shift, len(e) - min_shift, size=n)
    placebo = np.array([np.dot(np.roll(e, int(s)), r) for s in shifts])
    p = (1 + int((placebo >= actual).sum())) / (n + 1)
    return {"actual": actual, "placebo_mean": float(placebo.mean()), "placebo_sd": float(placebo.std()), "p_value": float(p)}


# ── verdict ────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class WindowStats:
    label: str
    months: int
    total_return: float
    hold_return: float
    mean_monthly: float
    mean_monthly_lower: float
    mean_exposure: float
    mean_excess: float
    mean_excess_lower: float
    placebo_p: float
    liquidations: int
    ruined: bool
    sufficient: bool


@dataclass(frozen=True)
class Verdict:
    hypothesis: str
    dev: WindowStats
    val: WindowStats
    alpha_adjusted: float
    absolute_ok: bool
    excess_ok: bool
    placebo_ok: bool

    @property
    def demonstrated(self) -> bool:
        return self.val.sufficient and self.absolute_ok and self.excess_ok and self.placebo_ok


def window_stats(
    label: str, result: BacktestResult, close: pd.Series, alpha: float, n_boot: int = 5000, seed: int = 42, block: int = 3
) -> WindowStats:
    """Statistiques d'une fenêtre. ``close`` : cours de clôture du sous-jacent sur la même fenêtre."""
    close = close.loc[result.equity.index[0] : result.equity.index[-1]]
    strat_m = monthly_returns(result.equity, result.initial_equity)
    hold_m = monthly_returns(close, float(close.iloc[0]))
    common = strat_m.index.intersection(hold_m.index)
    strat_m, hold_m = strat_m.loc[common], hold_m.loc[common]
    expo = float(result.exposure.mean())
    excess = strat_m.to_numpy() - expo * hold_m.to_numpy()
    sufficient = len(common) >= MIN_MONTHS

    def lb(x: np.ndarray) -> float:
        return lower_bound(x, alpha, n_boot, seed, block) if len(x) else float("nan")

    try:
        p = circular_shift_placebo(result.exposure.to_numpy(), result.bar_return.to_numpy(), seed=seed)["p_value"]
    except ValueError:
        p = float("nan")
    return WindowStats(
        label=label,
        months=len(common),
        total_return=result.total_return,
        hold_return=float(close.iloc[-1] / close.iloc[0] - 1.0),
        mean_monthly=float(strat_m.mean()) if len(strat_m) else float("nan"),
        mean_monthly_lower=lb(strat_m.to_numpy()),
        mean_exposure=expo,
        mean_excess=float(excess.mean()) if len(excess) else float("nan"),
        mean_excess_lower=lb(excess),
        placebo_p=p,
        liquidations=result.liquidations,
        ruined=result.ruined,
        sufficient=sufficient,
    )


def judge(
    hypothesis: str,
    dev_result: BacktestResult,
    val_result: BacktestResult,
    close: pd.Series,
    n_hypotheses: int = 1,
    alpha: float = 0.05,
    n_boot: int = 5000,
    seed: int = 42,
) -> Verdict:
    """Verdict sur VAL. DEV est affiché pour contexte et ne décide de rien."""
    adj = alpha / n_hypotheses
    dev = window_stats("DEV", dev_result, close, adj, n_boot, seed)
    val = window_stats("VAL", val_result, close, adj, n_boot, seed)
    return Verdict(
        hypothesis=hypothesis,
        dev=dev,
        val=val,
        alpha_adjusted=adj,
        absolute_ok=bool(val.mean_monthly_lower > 0),
        excess_ok=bool(val.mean_excess_lower > 0),
        placebo_ok=bool(val.placebo_p < adj),
    )
