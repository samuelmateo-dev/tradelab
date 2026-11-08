"""Ligne de commande : ``tradelab backtest`` et ``tradelab carry``."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from . import carry as carry_mod
from .backtest import run_backtest
from .costs import CostModel
from .data import load_csv, quality_report
from .strategy import StrategyConfig
from .validation import Verdict, judge

SAMPLE_BARS = "data/synthetic/eth_1h.csv.gz"      # données FICTIVES (voir synthetic.py)
SAMPLE_FUNDING = "data/synthetic/btc_funding.csv"


def _fake_data_note(path: Path) -> str:
    """Les fichiers de data/synthetic/ sont inventés : on le dit dans le rapport, pas seulement dans le README."""
    if "synthetic" in path.parts:
        return "ATTENTION : données FICTIVES (marche aléatoire). Ces résultats ne disent rien d'un vrai marché.\n"
    return ""


def fr(x: float, digits: int = 1, suffix: str = "") -> str:
    if x != x:  # NaN
        return "—"
    return f"{x:.{digits}f}".replace(".", ",") + suffix


def pct(x: float, digits: int = 1) -> str:
    return fr(x * 100, digits, " %")


def table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    line = lambda cells: "  " + "   ".join(c.ljust(widths[0]) if i == 0 else c.rjust(widths[i]) for i, c in enumerate(cells))
    return "\n".join([line(headers), "  " + "───".join("─" * w for w in widths), *map(line, rows)])


def _windows(bars: pd.DataFrame, split: str, cfg: StrategyConfig, costs: CostModel, equity: float):
    cut = pd.Timestamp(split, tz="UTC")
    dev = run_backtest(bars.loc[bars.index < cut], cfg, costs, equity)
    val = run_backtest(bars, cfg, costs, equity, start=cut)
    return dev, val


def cmd_backtest(a: argparse.Namespace) -> int:
    path = Path(a.data)
    if not path.exists():
        print(f"Fichier introuvable : {path}", file=sys.stderr)
        return 1
    bars = load_csv(path)
    q = quality_report(bars, "1h")
    costs = CostModel(taker_fee=a.fee, slippage=a.slippage, rollover_per_4h=a.rollover)
    hypotheses = [
        ("Bot tel quel (levier adaptatif ×2/×3/×5)", StrategyConfig()),
        ("Même stratégie sans levier (×1)", StrategyConfig(fixed_leverage=1)),
    ]
    verdicts: list[Verdict] = []
    for label, cfg in hypotheses:
        dev, val = _windows(bars, a.split, cfg, costs, a.equity)
        verdicts.append(judge(label, dev, val, bars["close"], n_hypotheses=len(hypotheses), n_boot=a.n_boot, seed=a.seed))

    if a.json:
        print(json.dumps([{**asdict(v), "demonstrated": v.demonstrated} for v in verdicts], indent=2, default=str))
        return 0

    print("tradelab — banc d'essai de la stratégie grille + short")
    print(_fake_data_note(path), end="")
    print(f"Données : {path} · {q.rows} bougies 1 h · {q.start:%Y-%m-%d} → {q.end:%Y-%m-%d}")
    print(f"Qualité : {q.missing_rows} heures manquantes ({pct(q.missing_share, 2)}), "
          f"{q.non_positive_prices} prix invalides, {q.inconsistent_ohlc} bougies incohérentes")
    print(f"Coûts supposés : frais {pct(a.fee, 2)}/ordre, glissement {pct(a.slippage, 3)}, rollover {pct(a.rollover, 3)}/4 h sur le montant emprunté")
    print("Hypothèses non vérifiées : rollover, liquidation à 40 % du niveau de marge (frais de liquidation 2 %)")
    print(f"Capital de départ {fr(a.equity, 0)} (nouveau compte au début de chaque fenêtre) · coupure DEV/VAL : {a.split}")
    print(f"Graine {a.seed} · {a.n_boot} tirages · α corrigé (Bonferroni, {len(hypotheses)} hypothèses) = {fr(verdicts[0].alpha_adjusted, 3)}\n")

    for v in verdicts:
        print(f"▌ {v.hypothesis}")
        rows = []
        for w in (v.dev, v.val):
            rows.append([
                w.label, str(w.months), pct(w.total_return, 0), pct(w.hold_return, 0), pct(w.mean_exposure, 0),
                pct(w.mean_monthly, 2), pct(w.mean_monthly_lower, 2), pct(w.mean_excess_lower, 2),
                fr(w.placebo_p, 3), str(w.liquidations),
            ])
        print(table(["Fenêtre", "Mois", "Stratégie", "Détention", "Expo moy.", "Moy. mensuelle", "Borne basse", "Excès (borne)", "p placebo", "Liq."], rows))
        print(f"  Conditions sur VAL → rentable : {'oui' if v.absolute_ok else 'non'} · bat la détention : {'oui' if v.excess_ok else 'non'} · "
              f"bat le placebo : {'oui' if v.placebo_ok else 'non'}")
        print(f"  VERDICT : {'DÉMONTRÉ' if v.demonstrated else 'non démontré'}"
              + ("" if v.val.sufficient else f"  (moins de 24 mois de validation)") + "\n")

    if a.sensitivity:
        print("▌ Sensibilité aux coûts (période complète, sans levier, capital neuf)")
        scenarios = [
            ("aucun coût", CostModel(0.0, 0.0, 0.0, 0.0)),
            ("frais 0,16 % + glissement", CostModel(taker_fee=0.0016, slippage=a.slippage)),
            ("frais 0,26 % + glissement", CostModel(taker_fee=0.0026, slippage=a.slippage)),
            ("frais 0,40 % + glissement", CostModel(taker_fee=0.0040, slippage=a.slippage)),
        ]
        rows = []
        for name, c in scenarios:
            r = run_backtest(bars, StrategyConfig(fixed_leverage=1), c, a.equity)
            rows.append([name, pct(r.total_return, 0), str(r.trades), fr(r.fees_paid, 1)])
        print(table(["Scénario", "Rendement", "Trades", "Frais payés"], rows))
        print()
    print("Rappel : exécution supposée à l'ouverture suivante, paramètres de coûts non mesurés sur de vrais ordres.")
    return 0


def cmd_carry(a: argparse.Namespace) -> int:
    path = Path(a.funding)
    if not path.exists():
        print(f"Fichier introuvable : {path}", file=sys.stderr)
        return 1
    s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    rows = carry_mod.yearly_table(s, a.friction)
    print("tradelab — portage du financement (jambe comptant + jambe perpétuel vendue), BTC")
    print(_fake_data_note(path), end="")
    print(f"Données : {path} · {len(s)} jours · {s.index[0]:%Y-%m-%d} → {s.index[-1]:%Y-%m-%d} · frictions supposées {pct(a.friction, 1)}/an\n")
    print(table(["Année", "Jours", "Brut /an", "Net /an", "Jours négatifs"],
                [[str(r.year), str(r.days), fr(r.gross_pct, 1, " %"), fr(r.net_pct, 1, " %"), pct(r.negative_day_share, 0)] for r in rows]))
    print()
    for days, lab in ((365, "12 derniers mois"), (180, "6 derniers mois"), (90, "3 derniers mois")):
        g, n = carry_mod.trailing_net_pct(s, days, a.friction)
        print(f"  {lab:18s} brut {fr(g, 1, ' %')}/an · net {fr(n, 1, ' %')}/an")
    worst, share = carry_mod.worst_window(s, 90, a.friction)
    print(f"  Pire fenêtre de 90 jours (net) : {pct(worst, 2)} · fenêtres négatives : {pct(share, 1)}")
    _, net12 = carry_mod.trailing_net_pct(s, 365, a.friction)
    print("\n  Rapporté au capital immobilisé (spot + marge du perpétuel), 12 derniers mois :")
    for lev in (1, 2, 3):
        print(f"    perpétuel à ×{lev} : {fr(carry_mod.yield_on_capital(net12, lev), 2, ' %')}/an")
    print("\n  Hors mesure : liquidation de la jambe vendue, risque de plateforme, écart perpétuel/comptant, impôts, accès au produit.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tradelab", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("backtest", help="Rejoue la stratégie grille + short et la soumet au juge statistique")
    b.add_argument("--data", default=SAMPLE_BARS, help="Bougies 1 h (CSV[.gz] : time, open, high, low, close, volume)")
    b.add_argument("--equity", type=float, default=50.0, help="Capital de départ (défaut 50)")
    b.add_argument("--split", default="2023-01-01", help="Début de la fenêtre de validation (défaut 2023-01-01)")
    b.add_argument("--fee", type=float, default=0.0026, help="Frais par ordre (défaut 0,0026)")
    b.add_argument("--slippage", type=float, default=0.0003, help="Glissement par ordre (défaut 0,0003)")
    b.add_argument("--rollover", type=float, default=0.0002, help="Rollover par 4 h sur le montant emprunté (hypothèse, défaut 0,0002)")
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--n-boot", type=int, default=5000)
    b.add_argument("--sensitivity", action="store_true", help="Ajoute le tableau de sensibilité aux coûts (≈ 15 s de plus)")
    b.add_argument("--json", action="store_true")
    b.set_defaults(func=cmd_backtest)

    c = sub.add_parser("carry", help="Mesure ce que le financement des perpétuels a rapporté (portage)")
    c.add_argument("--funding", default=SAMPLE_FUNDING)
    c.add_argument("--friction", type=float, default=0.02, help="Frictions annuelles retirées d'office (défaut 0,02)")
    c.set_defaults(func=cmd_carry)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
