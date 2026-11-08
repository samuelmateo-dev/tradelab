"""Modèle de coûts. Toutes les valeurs par défaut sont des HYPOTHÈSES à vérifier sur ton compte.

- ``taker_fee`` : 0,26 % est la valeur utilisée par le bot d'origine (``FEES``).
- ``slippage`` : écart entre le prix de la bougie et le prix obtenu. Non mesuré : choisi pour un
  petit ordre sur une paire liquide. À remplacer par une mesure sur de vrais ordres.
- ``rollover_per_4h`` : Kraken facture la marge sur le montant EMPRUNTÉ, à un taux variable fixé à
  l'exécution (page d'aide Kraken sur les frais de marge). Le taux et l'intervalle utilisés ici
  sont une hypothèse de travail, pas une donnée Kraken.
- ``liquidation_level`` : niveau de marge (équité / marge utilisée) sous lequel tout est liquidé.
  40 % est une HYPOTHÈSE de travail : la page d'aide Kraken consultée ne donne aucun pourcentage
  (le ``marge_utils.py`` d'origine affiche des niveaux 100 / 80 / 40 % dans un tableau « et si »,
  sans les présenter comme des seuils de l'exchange). ``margin_call_level`` n'est pas utilisé.
- ``liquidation_fee`` : 2 %, valeur donnée par la page d'aide Kraken sur la liquidation.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    taker_fee: float = 0.0026
    slippage: float = 0.0003
    rollover_per_4h: float = 0.0002
    liquidation_fee: float = 0.02
    margin_call_level: float = 0.80
    liquidation_level: float = 0.40

    def fee(self, notional: float) -> float:
        """Frais d'un ordre : proportionnels au NOTIONNEL (quantité × prix), pas au profit."""
        return self.taker_fee * abs(notional)

    def buy_price(self, mid: float) -> float:
        return mid * (1.0 + self.slippage)

    def sell_price(self, mid: float) -> float:
        return mid * (1.0 - self.slippage)

    def rollover(self, borrowed: float, hours: float) -> float:
        return self.rollover_per_4h * max(borrowed, 0.0) * hours / 4.0


FREE = CostModel(taker_fee=0.0, slippage=0.0, rollover_per_4h=0.0, liquidation_fee=0.0)
"""Coûts nuls : sert aux tests, où l'on veut isoler la logique du prix des frais."""


def true_round_trip_pnl(qty: float, entry: float, exit_: float, fee_rate: float, side: str = "long") -> float:
    """PnL net d'un aller-retour : gain brut moins les frais sur le notionnel d'entrée ET de sortie."""
    gross = qty * (exit_ - entry) if side == "long" else qty * (entry - exit_)
    return gross - fee_rate * (qty * entry + qty * exit_)


def legacy_reported_pnl(qty: float, entry: float, exit_: float, fee_rate: float, leverage: int, side: str = "long") -> float:
    """PnL tel que l'ancien ``bot.py`` l'affichait (lignes 550, 622, 721, 768).

    Deux défauts cumulés, conservés ici uniquement pour les mesurer :
    1. les frais sont retirés comme une fraction du PROFIT (``× (1 − 2·FEES)``) au lieu d'une
       fraction du notionnel ;
    2. le résultat est multiplié par le levier, alors que la quantité ``qty`` est déjà la
       quantité réellement exposée.
    """
    gross = qty * (exit_ - entry) if side == "long" else qty * (entry - exit_)
    return gross * (1.0 - 2.0 * fee_rate) * leverage
