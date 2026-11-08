"""Moteur de backtest : rejoue ``strategy.decide`` bougie par bougie sur un compte de simulation.

Chronologie d'une bougie *i* (aucune information du futur) :
1. les ordres décidés à la clôture de *i−1* s'exécutent à l'OUVERTURE de *i* (le bot réel agit
   quelques minutes après la clôture) ;
2. le rollover de la bougie est prélevé, puis la liquidation est testée au pire prix de la bougie ;
3. à la clôture de *i*, la stratégie décide ; ses ordres attendront l'ouverture de *i+1*.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .account import Fill, PaperAccount
from .costs import CostModel
from .strategy import (
    AccountView,
    Market,
    Order,
    StrategyConfig,
    StrategyState,
    decide,
    first_valid_index,
    prepare_features,
)


@dataclass(frozen=True)
class BacktestResult:
    equity: pd.Series            # équité NETTE à chaque clôture (frais de sortie de la dernière bougie inclus)
    exposure: pd.Series          # exposition nette / équité à l'ouverture de chaque bougie, après exécution
    gross_exposure: pd.Series
    bar_return: pd.Series        # close / open − 1 de chaque bougie
    initial_equity: float
    fills: list[Fill]
    rejected: int
    liquidations: int
    rollover_paid: float
    ruined: bool

    @property
    def total_return(self) -> float:
        return float(self.equity.iloc[-1] / self.initial_equity - 1.0)

    @property
    def trades(self) -> int:
        return sum(1 for f in self.fills if f.action == "close")

    @property
    def fees_paid(self) -> float:
        return float(sum(f.fee for f in self.fills))


def _execute(acct: PaperAccount, st: StrategyState, order: Order, mid: float, ts: pd.Timestamp) -> None:
    if order.action == "open_long":
        fill = acct.open_position("long", order.qty, mid, order.leverage, ts, order.reason)
    elif order.action == "open_short":
        fill = acct.open_position("short", order.qty, mid, order.leverage, ts, order.reason)
    elif order.action == "close_long":
        acct.close_position("long", order.qty, mid, ts, order.reason)
        return
    else:
        acct.close_position("short", order.qty, mid, ts, order.reason)
        return
    if fill is None:
        st.revert(order)


def run_backtest(
    bars: pd.DataFrame,
    cfg: StrategyConfig = StrategyConfig(),
    costs: CostModel = CostModel(),
    initial_equity: float = 50.0,
    bar_hours: float = 1.0,
    start: pd.Timestamp | str | None = None,
) -> BacktestResult:
    """Joue la stratégie sur ``bars`` (colonnes open, high, low, close, volume, index UTC).

    ``start`` : date à laquelle le compte démarre (capital neuf). Les bougies d'avant servent
    uniquement à amorcer les indicateurs, jamais à trader.
    """
    feats = prepare_features(bars)
    start = max(
        first_valid_index(feats),
        int(feats.index.searchsorted(pd.Timestamp(start, tz="UTC") if isinstance(start, str) else start))
        if start is not None
        else 0,
    )
    if start >= len(feats) - 2:
        raise ValueError("Historique trop court pour amorcer les indicateurs.")

    idx = feats.index
    o, h, l, c = (feats[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    cols = {k: feats[k].to_numpy(float) for k in ("adx", "atr_pct", "rsi", "rsi_prev", "ma50", "bbl", "bbu")}

    acct = PaperAccount(cash=initial_equity, costs=costs)
    st = StrategyState()
    pending: list[Order] = []

    n = len(feats) - start
    equity = np.full(n, np.nan)
    expo = np.zeros(n)
    gexpo = np.zeros(n)

    for k, i in enumerate(range(start, len(feats))):
        ts = idx[i]
        for order in pending:
            _execute(acct, st, order, o[i], ts)
        pending = []

        acct.accrue_rollover(bar_hours)
        if acct.check_liquidation(l[i], h[i], ts):
            st.reset_after_liquidation()

        expo[k] = acct.net_exposure(o[i])
        gexpo[k] = acct.gross_exposure(o[i])
        equity[k] = acct.equity(c[i])

        if acct.ruined or i == len(feats) - 1:
            continue
        m = Market(ts, c[i], h[i], l[i], cols["adx"][i], cols["atr_pct"][i], cols["rsi"][i],
                   cols["rsi_prev"][i], cols["ma50"][i], cols["bbl"][i], cols["bbu"][i])
        view = AccountView(
            equity=equity[k],
            free_margin=acct.free_margin(c[i]),
            long_qty=acct.long.qty if acct.long else 0.0,
            short_qty=acct.short.qty if acct.short else 0.0,
        )
        pending = decide(m, st, view, cfg)

    # Fermeture virtuelle des positions restantes, frais de sortie compris (sinon on gonfle le résultat).
    last_ts, last_close = idx[-1], c[-1]
    for side in ("long", "short"):
        acct.close_position(side, None, last_close, last_ts, "fin du backtest")
    equity[-1] = acct.cash

    sl = slice(start, None)
    return BacktestResult(
        equity=pd.Series(equity, index=idx[sl], name="equity"),
        exposure=pd.Series(expo, index=idx[sl], name="exposure"),
        gross_exposure=pd.Series(gexpo, index=idx[sl], name="gross_exposure"),
        bar_return=pd.Series(c[sl] / o[sl] - 1.0, index=idx[sl], name="bar_return"),
        initial_equity=initial_equity,
        fills=acct.fills,
        rejected=acct.rejected,
        liquidations=acct.liquidations,
        rollover_paid=acct.rollover_paid,
        ruined=acct.ruined,
    )
