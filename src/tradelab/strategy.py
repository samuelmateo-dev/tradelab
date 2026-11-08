"""Stratégie « grille + short + levier adaptatif », portée depuis ``bot.py`` en fonction pure.

``decide()`` ne fait AUCUNE entrée/sortie : elle lit un marché, un état et une vue du compte, puis
renvoie des ordres. Le même code sert au backtest et pourrait servir derrière un courtier réel.

Fidélité au bot d'origine (``bot.py``, ETH/EUR, bougies 1 h, bougie fermée) :
régime ADX/MA50, grille de 10 niveaux à 1,5 % ajustée par l'ATR, short sur retombée du RSI sous 65,
stop 3 %, objectif 3 %, stop suiveur par achat, niveau de levier selon le capital (×2 / ×3 / ×5).

Écarts assumés (listés dans le README) :
1. l'inventaire vendable est décrémenté pendant le cycle (``bot.py`` ne le fait pas : deux paliers
   déclenchés ensemble peuvent y vendre plus que ce qui est détenu) ;
2. la marge libre du compte remplace le solde EUR « free » de Kraken ;
3. les quantités sont plafonnées par le minimum d'ordre de l'exchange.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from .indicators import adx, atr, bollinger, rsi, sma

Regime = Literal["TREND_DOWN", "TREND_UP", "RANGE", "UNKNOWN"]
Action = Literal["open_long", "close_long", "open_short", "close_short"]


@dataclass(frozen=True)
class StrategyConfig:
    levels: int = 10
    spread: float = 0.015
    rebalance_hours: float = 24.0
    adx_trend: float = 20.0
    adx_short: float = 25.0
    rsi_short: float = 65.0
    rsi_cover: float = 35.0
    sl_short: float = 0.03
    tp_short: float = 0.03
    lev_thresholds: tuple[tuple[float, int], ...] = ((500.0, 5), (100.0, 3), (0.0, 2))
    fixed_leverage: int | None = None
    min_order_qty: float = 0.002
    min_short_qty: float = 0.01
    min_free_for_short: float = 10.0
    qty_decimals: int = 4


@dataclass(frozen=True)
class Market:
    """Photo du marché à la clôture d'une bougie."""
    ts: pd.Timestamp
    close: float
    high: float
    low: float
    adx: float
    atr_pct: float
    rsi: float
    rsi_prev: float
    ma50: float
    bbl: float
    bbu: float

    def regime(self, cfg: StrategyConfig) -> Regime:
        if self.adx >= cfg.adx_trend and self.close < self.ma50:
            return "TREND_DOWN"
        if self.adx >= cfg.adx_trend and self.close >= self.ma50:
            return "TREND_UP"
        return "RANGE"


@dataclass(frozen=True)
class AccountView:
    equity: float
    free_margin: float
    long_qty: float
    short_qty: float


@dataclass(frozen=True)
class Order:
    action: Action
    qty: float
    leverage: int
    reason: str
    ref: dict | None = field(default=None, compare=False, repr=False)


@dataclass
class StrategyState:
    grid: list[dict] = field(default_factory=list)
    center_price: float = 0.0
    last_rebalance: pd.Timestamp | None = None
    regime: Regime = "UNKNOWN"
    order_size: float = 0.0
    in_short: bool = False
    short_entry: float = 0.0
    buy_entries: list[dict] = field(default_factory=list)

    def revert(self, order: Order) -> None:
        """Annule les effets d'état d'un ordre que le compte a refusé (marge insuffisante)."""
        if order.action == "open_short":
            self.in_short = False
            self.short_entry = 0.0
        elif order.action == "open_long" and order.ref is not None:
            level = order.ref.get("level")
            if level is not None:
                level["filled"] = False
            self.buy_entries = [e for e in self.buy_entries if e is not order.ref.get("entry")]

    def reset_after_liquidation(self) -> None:
        self.in_short = False
        self.short_entry = 0.0
        self.buy_entries = []
        self.grid = []
        self.center_price = 0.0
        self.last_rebalance = None


# ── fonctions pures du bot d'origine ───────────────────────────────────────────
def get_leverage(equity: float, thresholds: tuple[tuple[float, int], ...]) -> int:
    for threshold, lev in thresholds:
        if equity >= threshold:
            return lev
    return 2


def adaptive_spread(atr_pct: float, base: float = 0.015) -> float:
    if atr_pct < 0.5:
        return max(base * 0.6, 0.007)
    if atr_pct < 1.0:
        return base * 0.8
    if atr_pct < 1.5:
        return base
    if atr_pct < 2.5:
        return base * 1.3
    return base * 1.6


def build_grid(center: float, levels: int, spread: float, regime: Regime) -> list[dict]:
    buy_n = int(levels * 0.7) if regime == "TREND_UP" else levels // 2
    sell_n = levels - buy_n
    grid = [{"price": round(center * (1 - spread * i), 2), "type": "BUY", "filled": False} for i in range(1, buy_n + 1)]
    grid += [{"price": round(center * (1 + spread * i), 2), "type": "SELL", "filled": False} for i in range(1, sell_n + 1)]
    grid.sort(key=lambda lvl: lvl["price"])
    return grid


def confidence_score(m: Market, signal: Literal["short", "grid_buy", "grid_sell"]) -> float:
    s = 0.0
    if signal == "short":
        s += 1.0 if m.adx >= 30 else 0.6 if m.adx >= 25 else 0.3 if m.adx >= 20 else 0.0
        s += 1.0 if m.rsi < 55 else 0.5 if m.rsi < 60 else 0.0
        if m.ma50 > 0:
            dist = (m.ma50 - m.close) / m.ma50 * 100
            s += 1.0 if dist > 2 else 0.5 if dist > 0.5 else 0.0
        s += 1.0 if 0.5 <= m.atr_pct <= 2.0 else 0.4 if m.atr_pct < 3.0 else 0.0
    elif signal == "grid_buy":
        s += 1.5 if m.bbl > 0 and m.close <= m.bbl else 0.7 if m.bbl > 0 and m.close <= m.bbl * 1.01 else 0.0
        s += 1.0 if m.rsi < 30 else 0.5 if m.rsi < 40 else 0.0
        s += 1.0 if m.atr_pct <= 1.5 else 0.4 if m.atr_pct <= 2.5 else 0.0
        s += 0.5 if m.adx < 20 else 0.0
    else:  # grid_sell
        s += 1.5 if m.bbu > 0 and m.close >= m.bbu else 0.7 if m.bbu > 0 and m.close >= m.bbu * 0.99 else 0.0
        s += 1.0 if m.rsi > 70 else 0.5 if m.rsi > 60 else 0.0
        s += 1.0 if m.atr_pct <= 1.5 else 0.4 if m.atr_pct <= 2.5 else 0.0
        s += 0.5 if m.adx < 20 else 0.0
    return min(s / 4.0, 1.0)


def confidence_to_size_pct(confidence: float, min_pct: float = 0.20, max_pct: float = 0.55) -> float:
    return min_pct + confidence * (max_pct - min_pct)


def order_size(equity: float, close: float, levels: int, leverage: int, cfg: StrategyConfig) -> float:
    return max(round(equity * leverage / levels / close, cfg.qty_decimals), cfg.min_order_qty)


# ── décision ───────────────────────────────────────────────────────────────────
def decide(m: Market, st: StrategyState, view: AccountView, cfg: StrategyConfig = StrategyConfig()) -> list[Order]:
    """Un cycle de décision. Modifie ``st`` en place et renvoie les ordres à exécuter."""
    orders: list[Order] = []
    regime = m.regime(cfg)
    leverage = cfg.fixed_leverage or get_leverage(view.equity, cfg.lev_thresholds)
    d = cfg.qty_decimals
    if st.in_short and view.short_qty <= 0:
        st.in_short = False            # le compte n'a plus de short (liquidation, refus) : on resynchronise
        st.short_entry = 0.0

    def qty(x: float, floor: float) -> float:
        return max(round(x, d), floor)

    if regime == "TREND_DOWN":
        if st.in_short:
            sl_price = st.short_entry * (1 + cfg.sl_short)
            tp_price = st.short_entry * (1 - cfg.tp_short)
            reason = None
            if m.high >= sl_price:
                reason = "stop-loss short"
            elif m.close <= tp_price:
                reason = "objectif short"
            elif m.rsi_prev >= cfg.rsi_cover and m.rsi < cfg.rsi_cover:
                reason = f"RSI<{cfg.rsi_cover:g}"
            if reason:
                orders.append(Order("close_short", view.short_qty, leverage, reason))
                st.in_short, st.short_entry = False, 0.0
        elif view.free_margin > cfg.min_free_for_short:
            if m.rsi_prev >= cfg.rsi_short and m.rsi < cfg.rsi_short and m.adx >= cfg.adx_short:
                conf = confidence_score(m, "short")
                size_pct = max(confidence_to_size_pct(conf, 0.50, 0.75), 0.50)
                q = qty(view.free_margin * size_pct / m.close, cfg.min_short_qty)
                orders.append(Order("open_short", q, leverage, "signal short"))
                st.in_short, st.short_entry = True, m.close
        if st.grid:
            st.grid, st.center_price, st.last_rebalance = [], 0.0, None
        return orders

    # RANGE ou TREND_UP : grille
    if st.in_short:
        orders.append(Order("close_short", view.short_qty, leverage, f"régime → {regime}"))
        st.in_short, st.short_entry = False, 0.0

    inv = view.long_qty                       # inventaire vendable, décrémenté au fil du cycle
    hours = (m.ts - st.last_rebalance).total_seconds() / 3600 if st.last_rebalance is not None else 9999.0

    if inv < st.order_size and view.free_margin > m.close * st.order_size * 2:
        orders.append(Order("open_long", qty(st.order_size * 3, cfg.min_order_qty), leverage, "auto-rééquilibrage"))
        st.last_rebalance, st.center_price = None, 0.0

    # `hours` est calculé AVANT l'auto-rééquilibrage, comme dans bot.py : la reconstruction forcée
    # (last_rebalance remis à None) ne prend effet qu'au cycle suivant.
    if not st.grid or st.regime != regime or hours >= cfg.rebalance_hours:
        spread = adaptive_spread(m.atr_pct, cfg.spread)
        conf = max(confidence_score(m, "grid_buy"), confidence_score(m, "grid_sell"))
        mult = confidence_to_size_pct(conf, 0.7, 1.3)
        st.grid = build_grid(m.close, cfg.levels, spread, regime)
        st.center_price, st.last_rebalance, st.regime = m.close, m.ts, regime
        st.order_size = order_size(view.equity, m.close, cfg.levels, leverage, cfg) * mult
        st.buy_entries = []

    size = qty(st.order_size, cfg.min_order_qty)

    st.buy_entries = [{"entry": e["entry"], "peak": max(e["peak"], m.close)} for e in st.buy_entries]
    trail_pct = adaptive_spread(m.atr_pct, cfg.spread)
    kept = []
    for e in st.buy_entries:
        if m.close <= e["peak"] * (1 - trail_pct) and inv >= st.order_size * 0.9:
            orders.append(Order("close_long", size, leverage, "stop suiveur"))
            inv -= size
        else:
            kept.append(e)
    st.buy_entries = kept

    for lvl in st.grid:
        if lvl["filled"]:
            continue
        if lvl["type"] == "BUY" and m.close <= lvl["price"]:
            entry = {"entry": m.close, "peak": m.close}
            orders.append(Order("open_long", size, leverage, "grille achat", ref={"level": lvl, "entry": entry}))
            lvl["filled"] = True
            st.buy_entries.append(entry)
            for sell in st.grid:
                if sell["type"] == "SELL" and sell["price"] > m.close:
                    sell["filled"] = False
                    break
        elif lvl["type"] == "SELL" and m.close >= lvl["price"]:
            if inv >= st.order_size * 0.9:
                orders.append(Order("close_long", size, leverage, "grille vente"))
                inv -= size
                lvl["filled"] = True
                for buy in st.grid:
                    if buy["type"] == "BUY" and buy["price"] < m.close:
                        buy["filled"] = False
                        break
            else:
                st.last_rebalance, st.center_price = None, 0.0   # inventaire insuffisant : reconstruction
    return orders


# ── préparation des indicateurs ────────────────────────────────────────────────
def prepare_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les indicateurs du bot d'origine (RSI 21, ADX 14, ATR 14, MA 50, Bollinger 14 / 2,5)."""
    out = bars.copy()
    out["rsi"] = rsi(out["close"], 21)
    out["rsi_prev"] = out["rsi"].shift(1)
    out["adx"] = adx(out["high"], out["low"], out["close"], 14)
    out["atr_pct"] = atr(out["high"], out["low"], out["close"], 14) / out["close"] * 100
    out["ma50"] = sma(out["close"], 50)
    out["bbl"], out["bbu"] = bollinger(out["close"], 14, 2.5)
    return out


def first_valid_index(features: pd.DataFrame) -> int:
    cols = ["rsi", "rsi_prev", "adx", "atr_pct", "ma50", "bbl", "bbu"]
    ok = features[cols].notna().all(axis=1).to_numpy()
    return int(np.argmax(ok)) if ok.any() else len(features)
