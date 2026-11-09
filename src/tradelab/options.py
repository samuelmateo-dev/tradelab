"""Ce que l'utilisateur peut choisir : paramètres de la stratégie, niveaux de levier à tester, analyses à lancer.

Tout est validé AVANT de lire les données ou de lancer un calcul : une faute de frappe, une valeur hors plage ou un nom
inconnu est refusé avec un message qui désigne l'entrée fautive. Rien n'est ignoré en silence.

Ce qui reste volontairement figé : la logique de décision de la stratégie (``strategy.decide``), la coupure
DEV/VAL, la durée minimale de validation (24 mois), le bootstrap par blocs de mois et le placebo. Les paramètres
ci-dessous règlent la stratégie ; ils ne changent pas la façon de la juger.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from .strategy import StrategyConfig

ANALYSES = ("verdict", "sensitivity")
ANALYSES_HELP = {
    "verdict": "rejoue la stratégie sur DEV puis VAL et rend le verdict statistique (rentable ? bat la détention ? bat le placebo ?)",
    "sensitivity": "refait la stratégie sans levier avec plusieurs niveaux de frais, sur la période complète (≈ 15 s de plus)",
}
MAX_FIXED_LEVERAGE = 10
# Champs de StrategyConfig qu'on ne règle PAS ici : le levier se choisit avec --leverage, pas avec --set.
LEVERAGE_FIELDS = ("fixed_leverage", "lev_thresholds")


class ConfigError(ValueError):
    """Option invalide : le message dit laquelle et pourquoi."""


@dataclass(frozen=True)
class Param:
    name: str
    kind: str   # "int" ou "float"
    low: float
    high: float
    meaning: str


PARAMS: tuple[Param, ...] = (
    Param("levels", "int", 2, 50, "nombre de niveaux de la grille (moitié achats, moitié ventes)"),
    Param("spread", "float", 0.001, 0.2, "écart entre niveaux (0,015 = 1,5 %), ensuite ajusté par la volatilité (ATR)"),
    Param("rebalance_hours", "float", 1, 720, "heures entre deux reconstructions de la grille"),
    Param("adx_trend", "float", 0, 100, "ADX à partir duquel le marché est « en tendance »"),
    Param("adx_short", "float", 0, 100, "ADX minimal pour ouvrir un short"),
    Param("rsi_short", "float", 1, 99, "le RSI doit retomber sous ce niveau pour ouvrir un short"),
    Param("rsi_cover", "float", 1, 99, "quand le RSI retombe sous ce niveau, on couvre le short (doit rester < rsi_short)"),
    Param("sl_short", "float", 0.001, 0.5, "stop du short (0,03 = 3 %)"),
    Param("tp_short", "float", 0.001, 0.5, "objectif du short (0,03 = 3 %)"),
    Param("min_order_qty", "float", 1e-6, 1000, "quantité minimale d'un ordre de grille"),
    Param("min_short_qty", "float", 1e-6, 1000, "quantité minimale d'un short"),
    Param("min_free_for_short", "float", 0, 1e6, "marge libre minimale pour ouvrir un short"),
    Param("qty_decimals", "int", 0, 8, "décimales de quantité acceptées par la plateforme"),
)
BY_NAME = {p.name: p for p in PARAMS}

EXAMPLE_TEXT = """{
  "levels": 8,
  "spread": 0.02,
  "sl_short": 0.04
}
"""


def _number(name: str, value: Any, param: Param) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} : nombre attendu, reçu {json.dumps(value, ensure_ascii=False)}")
    if isinstance(value, float) and not math.isfinite(value):
        raise ConfigError(f"{name} : valeur non finie")
    if param.kind == "int":
        if isinstance(value, float):
            raise ConfigError(f"{name} : entier attendu, reçu {value}")
        value = int(value)
    else:
        value = float(value)
    if not param.low <= value <= param.high:
        raise ConfigError(f"{name} : {value} hors de la plage acceptée [{_plain(param.low)} ; {_plain(param.high)}]")
    return value


def build_config(overrides: Mapping[str, Any] | None = None, base: StrategyConfig | None = None) -> StrategyConfig:
    """Paramètres du bot d'origine, sauf ceux de ``overrides`` (noms de ``PARAMS``)."""
    base = base or StrategyConfig()
    overrides = dict(overrides or {})
    changes: dict[str, Any] = {}
    for name, value in overrides.items():
        if name in LEVERAGE_FIELDS:
            raise ConfigError(f"{name} : le levier se choisit avec --leverage (par exemple --leverage adaptive,1,2), pas avec les paramètres")
        if name not in BY_NAME:
            raise ConfigError(f"{name} : paramètre inconnu (paramètres acceptés : {', '.join(BY_NAME)})")
        changes[name] = _number(name, value, BY_NAME[name])
    cfg = replace(base, **changes)
    if cfg.rsi_cover >= cfg.rsi_short:
        raise ConfigError(f"rsi_cover ({cfg.rsi_cover:g}) doit être inférieur à rsi_short ({cfg.rsi_short:g})")
    return cfg


def load_params(path: Path) -> dict[str, Any]:
    """Fichier JSON : un objet {nom: valeur}."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"fichier illisible : {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"JSON invalide : {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError('le fichier doit contenir un objet JSON, par exemple {"spread": 0.02}')
    return data


def parse_set(items: list[str]) -> dict[str, Any]:
    """``--set nom=valeur`` (virgule décimale acceptée : ``spread=0,02``). Un nom répété est refusé."""
    out: dict[str, Any] = {}
    for item in items:
        name, _, raw = item.partition("=")
        name, raw = name.strip(), raw.strip()
        if not name or not raw:  # sans « = », raw est vide : ce cas est couvert
            raise ConfigError(f"--set {item!r} : attendu nom=valeur, par exemple spread=0.02")
        if name in out:
            raise ConfigError(f"{name} : défini deux fois avec --set")
        param = BY_NAME.get(name)
        text = raw.replace(",", ".") if raw.count(",") == 1 and "." not in raw else raw
        if param is not None and param.kind == "int":
            try:
                out[name] = int(text)
            except ValueError:
                raise ConfigError(f"{name} : entier attendu, reçu {raw!r}") from None
        else:
            try:
                out[name] = float(text)
            except ValueError:
                if param is None:
                    out[name] = raw  # nom inconnu : build_config le refusera avec la liste des noms valides
                else:
                    raise ConfigError(f"{name} : nombre attendu, reçu {raw!r}") from None
    return out


def merge_params(file_params: Mapping[str, Any] | None, set_params: Mapping[str, Any] | None) -> dict[str, Any]:
    """Le fichier d'abord, puis --set par-dessus."""
    return {**(file_params or {}), **(set_params or {})}


def changed(cfg: StrategyConfig) -> dict[str, Any]:
    """Paramètres réglables qui diffèrent du bot d'origine."""
    default = StrategyConfig()
    return {p.name: getattr(cfg, p.name) for p in PARAMS if getattr(cfg, p.name) != getattr(default, p.name)}


def describe_changes(cfg: StrategyConfig) -> str:
    return ", ".join(f"{k}={v:g}" for k, v in changed(cfg).items())


@dataclass(frozen=True)
class Variant:
    """Une hypothèse testée : la stratégie réglée, avec un niveau de levier."""
    fixed_leverage: int | None   # None : levier adaptatif du bot (×2/×3/×5 selon le capital)
    config: StrategyConfig

    @property
    def label(self) -> str:
        modified = bool(changed(self.config))
        if self.fixed_leverage is None:
            text = "Stratégie modifiée (levier adaptatif ×2/×3/×5)" if modified else "Bot tel quel (levier adaptatif ×2/×3/×5)"
        elif self.fixed_leverage == 1:
            text = "Même stratégie sans levier (×1)"
        else:
            text = f"Même stratégie, levier fixe ×{self.fixed_leverage}"
        return f"{text} — {describe_changes(self.config)}" if modified else text


def parse_leverage(text: str, cfg: StrategyConfig) -> list[Variant]:
    """``adaptive,1,2`` -> trois hypothèses. Chaque entrée est le niveau de levier d'une hypothèse."""
    parts = [part.strip().lower() for part in text.split(",") if part.strip()]
    if not parts:
        raise ConfigError("--leverage : au moins un niveau (adaptive, 1, 2, ...)")
    variants: list[Variant] = []
    seen: set[str] = set()
    for part in parts:
        if part in seen:
            raise ConfigError(f"--leverage : « {part} » est donné deux fois")
        seen.add(part)
        if part == "adaptive":
            variants.append(Variant(None, replace(cfg, fixed_leverage=None)))
            continue
        try:
            level = int(part)
        except ValueError:
            raise ConfigError(f"--leverage : « {part} » invalide (adaptive, ou un entier entre 1 et {MAX_FIXED_LEVERAGE})") from None
        if not 1 <= level <= MAX_FIXED_LEVERAGE:
            raise ConfigError(f"--leverage : ×{level} hors de la plage [1 ; {MAX_FIXED_LEVERAGE}]")
        variants.append(Variant(level, replace(cfg, fixed_leverage=level)))
    return variants


def parse_analyses(text: str, sensitivity_flag: bool = False) -> list[str]:
    """``verdict,sensitivity``. ``--sensitivity`` (ancienne option) ajoute « sensitivity »."""
    names = [part.strip().lower() for part in text.split(",") if part.strip()]
    if not names:
        raise ConfigError(f"--analyses : au moins une analyse ({', '.join(ANALYSES)})")
    for name in names:
        if name not in ANALYSES:
            raise ConfigError(f"--analyses : « {name} » inconnue (analyses : {', '.join(ANALYSES)})")
    if len(set(names)) != len(names):
        raise ConfigError("--analyses : une analyse est donnée deux fois")
    if sensitivity_flag and "sensitivity" not in names:
        names.append("sensitivity")
    return [name for name in ANALYSES if name in names]  # ordre fixe : verdict d'abord


def check_tests_counted(counted: int | None, launched: int) -> int:
    """Nombre de tests pour la correction de Bonferroni : ceux lancés, ou ceux qu'on déclare (jamais moins)."""
    if counted is None:
        return launched
    if counted < launched:
        raise ConfigError(f"--tests-counted {counted} : au moins {launched} (les hypothèses lancées ici comptent déjà)")
    return counted


def check_alpha(alpha: float) -> float:
    if not 0 < alpha < 1:  # nan et inf échouent aussi à cette comparaison
        raise ConfigError(f"--alpha {alpha} : un seuil entre 0 et 1 est attendu (0,05 par défaut)")
    return alpha


def _plain(x: float) -> str:
    """Nombre sans notation scientifique : 1e-06 -> 0.000001, 1e+06 -> 1000000."""
    return f"{x:.6f}".rstrip("0").rstrip(".")


def defaults_table() -> list[tuple[str, str, str, str]]:
    """(nom, défaut, plage, sens) pour ``tradelab params``."""
    default = StrategyConfig()
    return [(p.name, f"{getattr(default, p.name):g}", f"{_plain(p.low)} – {_plain(p.high)}", p.meaning) for p in PARAMS]

