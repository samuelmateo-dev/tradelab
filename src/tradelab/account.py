"""Compte de simulation avec marge, rollover et liquidation.

Hypothèse centrale, héritée du bot d'origine : une position LONGUE et une position COURTE peuvent
coexister sur la même paire (mode « hedge »). Un vrai compte de marge Kraken les compense : les
scripts de réparation du projet d'origine (``close_leftover_xrp.py``, ``fix_after_incident.py``)
racontent précisément ce qui arrive quand le bot croit à deux positions indépendantes. Ce
simulateur reflète donc ce que le bot *croyait*, pas ce que l'exchange aurait fait.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from .costs import CostModel

Side = Literal["long", "short"]


@dataclass
class Position:
    side: Side
    qty: float
    entry: float           # prix d'entrée moyen pondéré
    margin: float          # marge bloquée, en monnaie de cotation
    opened_at: pd.Timestamp

    def unrealized(self, price: float) -> float:
        return self.qty * (price - self.entry) if self.side == "long" else self.qty * (self.entry - price)

    @property
    def borrowed(self) -> float:
        """Montant emprunté : ce que Kraken finance, donc la base du rollover."""
        notional = self.qty * self.entry
        borrowed = notional - self.margin
        return borrowed if borrowed > 1e-9 * notional else 0.0   # ignore le bruit d'arrondi à ×1


@dataclass(frozen=True)
class Fill:
    ts: pd.Timestamp
    action: Literal["open", "close", "liquidation"]
    side: Side
    qty: float
    price: float
    fee: float
    pnl: float          # PnL réalisé (hors frais) ; 0 à l'ouverture
    reason: str


@dataclass
class PaperAccount:
    cash: float
    costs: CostModel = field(default_factory=CostModel)
    long: Position | None = None
    short: Position | None = None
    fills: list[Fill] = field(default_factory=list)
    rejected: int = 0
    liquidations: int = 0
    rollover_paid: float = 0.0
    ruined: bool = False

    # ── lecture de l'état ─────────────────────────────────────────────
    def _positions(self) -> list[Position]:
        return [p for p in (self.long, self.short) if p is not None]

    def position(self, side: Side) -> Position | None:
        return self.long if side == "long" else self.short

    def equity(self, price: float) -> float:
        return self.cash + sum(p.unrealized(price) for p in self._positions())

    def margin_used(self) -> float:
        return sum(p.margin for p in self._positions())

    def free_margin(self, price: float) -> float:
        return self.equity(price) - self.margin_used()

    def margin_level(self, price: float) -> float:
        used = self.margin_used()
        return float("inf") if used <= 0 else self.equity(price) / used

    def net_exposure(self, price: float) -> float:
        """Exposition nette signée / équité (+1 = 100 % long, −1 = 100 % short)."""
        eq = self.equity(price)
        if eq <= 0:
            return 0.0
        signed = (self.long.qty if self.long else 0.0) - (self.short.qty if self.short else 0.0)
        return signed * price / eq

    def gross_exposure(self, price: float) -> float:
        eq = self.equity(price)
        if eq <= 0:
            return 0.0
        qty = (self.long.qty if self.long else 0.0) + (self.short.qty if self.short else 0.0)
        return qty * price / eq

    # ── ordres ────────────────────────────────────────────────────────
    def open_position(self, side: Side, qty: float, mid: float, leverage: int, ts: pd.Timestamp, reason: str = "") -> Fill | None:
        """Ouvre ou renforce une position. Refus (None) si la marge libre ne couvre pas marge + frais."""
        if qty <= 0 or leverage < 1:
            raise ValueError("Quantité positive et levier ≥ 1 requis.")
        price = self.costs.buy_price(mid) if side == "long" else self.costs.sell_price(mid)
        notional = qty * price
        margin = notional / leverage
        fee = self.costs.fee(notional)
        if self.free_margin(mid) < margin + fee:
            self.rejected += 1
            return None
        self.cash -= fee
        pos = self.position(side)
        if pos is None:
            pos = Position(side, qty, price, margin, ts)
            if side == "long":
                self.long = pos
            else:
                self.short = pos
        else:
            total = pos.qty + qty
            pos.entry = (pos.entry * pos.qty + price * qty) / total
            pos.qty = total
            pos.margin += margin
        fill = Fill(ts, "open", side, qty, price, fee, 0.0, reason)
        self.fills.append(fill)
        return fill

    def close_position(self, side: Side, qty: float | None, mid: float, ts: pd.Timestamp, reason: str = "") -> Fill | None:
        """Ferme tout ou partie d'une position. ``qty=None`` ferme tout. Aucune position : None."""
        pos = self.position(side)
        if pos is None:
            return None
        q = pos.qty if qty is None else min(qty, pos.qty)
        if q <= 0:
            return None
        price = self.costs.sell_price(mid) if side == "long" else self.costs.buy_price(mid)
        pnl = q * (price - pos.entry) if side == "long" else q * (pos.entry - price)
        fee = self.costs.fee(q * price)
        self.cash += pnl - fee
        share = q / pos.qty
        pos.margin *= 1.0 - share
        pos.qty -= q
        if pos.qty <= 1e-12:
            if side == "long":
                self.long = None
            else:
                self.short = None
        fill = Fill(ts, "close", side, q, price, fee, pnl, reason)
        self.fills.append(fill)
        return fill

    # ── passage du temps ──────────────────────────────────────────────
    def accrue_rollover(self, hours: float) -> float:
        total = sum(self.costs.rollover(p.borrowed, hours) for p in self._positions())
        self.cash -= total
        self.rollover_paid += total
        return total

    def check_liquidation(self, low: float, high: float, ts: pd.Timestamp) -> bool:
        """Liquide tout si, au plus mauvais prix de la bougie, l'équité tombe sous le seuil de marge.

        Approximation pessimiste : liquidation au pire prix de la bougie, avec les frais de liquidation.
        """
        if not self._positions():
            return False
        used = self.margin_used()
        worst_price = min((low, high), key=lambda p: self.equity(p))
        if self.equity(worst_price) > self.costs.liquidation_level * used:
            return False
        notional = sum(p.qty * worst_price for p in self._positions())
        fee = self.costs.liquidation_fee * notional
        for p in self._positions():
            pnl = p.unrealized(worst_price)
            self.cash += pnl
            self.fills.append(Fill(ts, "liquidation", p.side, p.qty, worst_price, 0.0, pnl, "liquidation"))
        self.cash -= fee
        self.long = self.short = None
        self.liquidations += 1
        if self.cash <= 0:
            self.cash = 0.0
            self.ruined = True
        return True
