"""Portage « delta-neutre » : acheter au comptant et vendre un perpétuel de même taille, encaisser le financement.

Le prix s'annule entre les deux jambes ; ne reste que le financement que les acheteurs de perpétuels
versent (ou reçoivent de) leurs vendeurs. Ce module mesure ce que ce financement a rapporté, année par
année, et non ce qu'il rapportera.

Ce que la mesure NE couvre PAS : liquidation de la jambe vendue en cas de hausse brutale, risque de la
plateforme, écart perpétuel/comptant, impôts, produit disponible sur ton compte et dans ton pays, et
le fait que le financement de Kraken n'est pas celui de Binance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DAYS = 365


@dataclass(frozen=True)
class CarryYear:
    year: int
    days: int
    gross_pct: float          # rendement annualisé sur le notionnel, avant frictions
    net_pct: float            # après frictions annuelles
    negative_day_share: float


def daily_friction(annual_friction: float) -> float:
    return annual_friction / DAYS


def yearly_table(funding: pd.Series, annual_friction: float = 0.02) -> list[CarryYear]:
    """``funding`` : somme quotidienne du financement, en part du notionnel (positif = le vendeur reçoit)."""
    if funding.empty:
        raise ValueError("Série de financement vide.")
    fr = daily_friction(annual_friction)
    rows = []
    for year, g in funding.groupby(funding.index.year):
        n = len(g)
        rows.append(
            CarryYear(
                year=int(year),
                days=n,
                gross_pct=float(g.sum() * DAYS / n * 100),
                net_pct=float((g - fr).sum() * DAYS / n * 100),
                negative_day_share=float((g < 0).mean()),
            )
        )
    return rows


def trailing_net_pct(funding: pd.Series, days: int, annual_friction: float = 0.02) -> tuple[float, float]:
    """(brut, net) annualisés sur les ``days`` derniers jours."""
    s = funding.iloc[-days:]
    n = len(s)
    fr = daily_friction(annual_friction)
    return float(s.sum() * DAYS / n * 100), float((s - fr).sum() * DAYS / n * 100)


def worst_window(funding: pd.Series, days: int = 90, annual_friction: float = 0.02) -> tuple[float, float]:
    """(pire, part de fenêtres négatives) du cumul net sur des fenêtres glissantes, en part du notionnel."""
    net = funding.to_numpy(dtype=float) - daily_friction(annual_friction)
    if len(net) < days:
        raise ValueError("Série plus courte que la fenêtre demandée.")
    cs = np.concatenate([[0.0], np.cumsum(net)])
    w = cs[days:] - cs[:-days]
    return float(w.min()), float((w < 0).mean())


def yield_on_capital(notional_yield: float, perp_leverage: float = 1.0) -> float:
    """Rendement rapporté au capital réellement immobilisé : spot (1 unité) + marge du perpétuel (1/levier)."""
    if perp_leverage < 1:
        raise ValueError("Levier du perpétuel ≥ 1.")
    return notional_yield / (1.0 + 1.0 / perp_leverage)
