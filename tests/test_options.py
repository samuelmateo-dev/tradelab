"""Ce que l'utilisateur peut choisir (paramètres, leviers, analyses) : validation stricte, rien d'ignoré en silence."""
import json
import math
from dataclasses import fields
from pathlib import Path

import pytest

from tradelab import options
from tradelab.options import ConfigError, PARAMS, build_config, parse_analyses, parse_leverage, parse_set
from tradelab.strategy import StrategyConfig

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_FILE = ROOT / "examples" / "params.example.json"


# --- Le catalogue des paramètres ----------------------------------------------------------------------

def test_every_strategy_field_is_either_adjustable_or_reserved_for_leverage():
    """Un champ ajouté plus tard à StrategyConfig oblige à décider s'il est réglable."""
    assert {f.name for f in fields(StrategyConfig)} == {p.name for p in PARAMS} | set(options.LEVERAGE_FIELDS)
    assert len({p.name for p in PARAMS}) == len(PARAMS)


@pytest.mark.parametrize("param", PARAMS, ids=lambda p: p.name)
def test_defaults_are_inside_their_range_and_the_range_edges_are_accepted(param):
    default = getattr(StrategyConfig(), param.name)
    assert param.low <= default <= param.high
    cast = int if param.kind == "int" else float
    low, high = cast(param.low), cast(param.high)
    overrides = {param.name: low}
    if param.name == "rsi_short":
        overrides = {param.name: low, "rsi_cover": 1} if low > 1 else {param.name: 2, "rsi_cover": 1}
    assert getattr(build_config(overrides), param.name) == overrides[param.name]
    if param.name == "rsi_cover":
        return  # la borne haute de rsi_cover est contrainte par rsi_short : testé à part
    assert getattr(build_config({param.name: high}), param.name) == high


@pytest.mark.parametrize("param", PARAMS, ids=lambda p: p.name)
def test_values_just_outside_the_range_are_refused(param):
    below = param.low - 1 if param.kind == "int" else math.nextafter(param.low, -math.inf)
    above = param.high + 1 if param.kind == "int" else math.nextafter(param.high, math.inf)
    for value in (below, above):
        with pytest.raises(ConfigError, match=rf"{param.name} : .* hors de la plage acceptée"):
            build_config({param.name: value})


def test_no_override_means_the_original_bot():
    assert build_config() == StrategyConfig() == build_config({})


def test_overrides_are_applied_and_the_rest_is_untouched():
    cfg = build_config({"levels": 8, "spread": 0.02})
    assert (cfg.levels, cfg.spread) == (8, 0.02)
    assert cfg == StrategyConfig(levels=8, spread=0.02)


def test_an_int_is_accepted_for_a_float_parameter():
    cfg = build_config({"adx_trend": 25})
    assert cfg.adx_trend == 25.0 and isinstance(cfg.adx_trend, float)


@pytest.mark.parametrize("value, message", [
    (True, r"nombre attendu, reçu true"),
    ("8", r"nombre attendu, reçu \"8\""),
    (None, r"nombre attendu, reçu null"),
    ([8], r"nombre attendu"),
    (8.0, r"entier attendu, reçu 8\.0"),
    (8.5, r"entier attendu"),
    (float("nan"), r"valeur non finie"),
    (float("inf"), r"valeur non finie"),
])
def test_wrong_value_types_are_refused(value, message):
    with pytest.raises(ConfigError, match=rf"levels : {message}"):
        build_config({"levels": value})


def test_unknown_parameter_lists_the_valid_names():
    with pytest.raises(ConfigError, match=r"spred : paramètre inconnu \(paramètres acceptés : levels, spread, rebalance_hours"):
        build_config({"spred": 0.02})


@pytest.mark.parametrize("name", options.LEVERAGE_FIELDS)
def test_leverage_is_not_a_parameter(name):
    with pytest.raises(ConfigError, match=rf"{name} : le levier se choisit avec --leverage"):
        build_config({name: 2})


def test_rsi_cover_must_stay_below_rsi_short():
    with pytest.raises(ConfigError, match=r"rsi_cover \(70\) doit être inférieur à rsi_short \(65\)"):
        build_config({"rsi_cover": 70})
    with pytest.raises(ConfigError, match=r"rsi_cover \(35\) doit être inférieur à rsi_short \(30\)"):
        build_config({"rsi_short": 30})
    with pytest.raises(ConfigError, match="doit être inférieur"):
        build_config({"rsi_cover": 50, "rsi_short": 50})  # l'égalité est refusée aussi
    assert build_config({"rsi_cover": 40, "rsi_short": 41}).rsi_cover == 40


# --- --set et --params ---------------------------------------------------------------------------------

def test_parse_set_reads_numbers_and_accepts_a_decimal_comma():
    assert parse_set(["spread=0,02", "levels=8", "sl_short=0.04"]) == {"spread": 0.02, "levels": 8, "sl_short": 0.04}
    assert isinstance(parse_set(["levels=8"])["levels"], int)
    assert parse_set([" spread = 0.02 "]) == {"spread": 0.02}


@pytest.mark.parametrize("item, message", [
    ("spread", r"attendu nom=valeur"),
    ("=0.02", r"attendu nom=valeur"),
    ("spread=", r"attendu nom=valeur"),
    ("levels=8.0", r"levels : entier attendu, reçu '8.0'"),
    ("levels=huit", r"levels : entier attendu"),
    ("spread=beaucoup", r"spread : nombre attendu, reçu 'beaucoup'"),
    ("spread=0,0,2", r"spread : nombre attendu"),
    ("spread=1,000.5", r"spread : nombre attendu"),
])
def test_parse_set_rejects_malformed_items(item, message):
    with pytest.raises(ConfigError, match=message):
        parse_set([item])


def test_parse_set_refuses_a_name_given_twice():
    with pytest.raises(ConfigError, match=r"spread : défini deux fois avec --set"):
        parse_set(["spread=0.02", "spread=0.03"])


def test_an_unknown_name_in_set_ends_up_refused_with_the_list_of_valid_names():
    for item in ("bogus=1", "bogus=text"):
        with pytest.raises(ConfigError, match=r"bogus : paramètre inconnu"):
            build_config(parse_set([item]))


def test_set_overrides_the_file():
    merged = options.merge_params({"spread": 0.02, "levels": 8}, {"spread": 0.03})
    assert merged == {"spread": 0.03, "levels": 8}
    assert options.merge_params(None, None) == {}


def test_load_params_reports_unreadable_malformed_and_non_object_files(tmp_path):
    with pytest.raises(ConfigError, match="illisible"):
        options.load_params(tmp_path / "absent.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{ pas du json", encoding="utf-8")
    with pytest.raises(ConfigError, match="JSON invalide"):
        options.load_params(bad)
    listing = tmp_path / "liste.json"
    listing.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ConfigError, match="objet JSON"):
        options.load_params(listing)


def test_example_file_matches_the_text_shown_by_tradelab_params():
    assert EXAMPLE_FILE.read_text(encoding="utf-8") == options.EXAMPLE_TEXT
    cfg = build_config(options.load_params(EXAMPLE_FILE))
    assert (cfg.levels, cfg.spread, cfg.sl_short) == (8, 0.02, 0.04)


# --- Hypothèses : leviers ------------------------------------------------------------------------------

def test_labels_of_the_default_hypotheses_are_the_original_ones():
    adaptive, unlevered = parse_leverage("adaptive,1", StrategyConfig())
    assert adaptive.label == "Bot tel quel (levier adaptatif ×2/×3/×5)"
    assert unlevered.label == "Même stratégie sans levier (×1)"
    assert adaptive.config == StrategyConfig() and unlevered.config == StrategyConfig(fixed_leverage=1)


def test_labels_say_when_the_strategy_was_modified():
    cfg = build_config({"spread": 0.02})
    adaptive, one, three = parse_leverage("adaptive,1,3", cfg)
    assert adaptive.label == "Stratégie modifiée (levier adaptatif ×2/×3/×5) — spread=0.02"
    assert one.label == "Même stratégie sans levier (×1) — spread=0.02"
    assert three.label == "Même stratégie, levier fixe ×3 — spread=0.02"
    assert [v.fixed_leverage for v in (adaptive, one, three)] == [None, 1, 3]
    assert three.config == StrategyConfig(spread=0.02, fixed_leverage=3)


def test_setting_a_parameter_to_its_default_is_not_a_modification():
    cfg = build_config({"spread": 0.015})
    assert options.changed(cfg) == {} and options.describe_changes(cfg) == ""
    assert parse_leverage("adaptive", cfg)[0].label.startswith("Bot tel quel")


def test_leverage_list_is_case_and_space_tolerant_and_the_maximum_is_accepted():
    variants = parse_leverage(" ADAPTIVE , 10 ", StrategyConfig())
    assert [v.fixed_leverage for v in variants] == [None, 10]


@pytest.mark.parametrize("text, message", [
    ("", r"au moins un niveau"),
    (" , ", r"au moins un niveau"),
    ("0", r"×0 hors de la plage \[1 ; 10\]"),
    ("11", r"×11 hors de la plage"),
    ("-2", r"hors de la plage"),
    ("2.5", r"« 2\.5 » invalide"),
    ("x", r"« x » invalide \(adaptive, ou un entier"),
    ("adaptive,adaptive", r"« adaptive » est donné deux fois"),
    ("1,2,1", r"« 1 » est donné deux fois"),
])
def test_invalid_leverage_lists_are_refused(text, message):
    with pytest.raises(ConfigError, match=message):
        parse_leverage(text, StrategyConfig())


# --- Analyses, alpha, nombre de tests -----------------------------------------------------------------

def test_analyses_parsing_and_fixed_order():
    assert parse_analyses("verdict") == ["verdict"]
    assert parse_analyses("sensitivity,verdict") == ["verdict", "sensitivity"]
    assert parse_analyses("sensitivity") == ["sensitivity"]
    assert parse_analyses("verdict", sensitivity_flag=True) == ["verdict", "sensitivity"]
    assert parse_analyses("sensitivity", sensitivity_flag=True) == ["sensitivity"]
    assert parse_analyses(" Verdict ") == ["verdict"]


@pytest.mark.parametrize("text, message", [
    ("", r"au moins une analyse"),
    ("foo", r"« foo » inconnue \(analyses : verdict, sensitivity\)"),
    ("verdict,verdict", r"donnée deux fois"),
])
def test_invalid_analyses_are_refused(text, message):
    with pytest.raises(ConfigError, match=message):
        parse_analyses(text)


def test_tests_counted_can_only_add_to_the_hypotheses_launched():
    assert options.check_tests_counted(None, 2) == 2
    assert options.check_tests_counted(2, 2) == 2
    assert options.check_tests_counted(10, 2) == 10
    with pytest.raises(ConfigError, match=r"--tests-counted 1 : au moins 2"):
        options.check_tests_counted(1, 2)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5, float("nan"), float("inf")])
def test_alpha_must_be_strictly_between_zero_and_one(alpha):
    with pytest.raises(ConfigError, match="seuil entre 0 et 1"):
        options.check_alpha(alpha)


def test_alpha_accepts_usual_values():
    assert options.check_alpha(0.05) == 0.05 and options.check_alpha(0.999) == 0.999 and options.check_alpha(0.001) == 0.001


def test_displayed_ranges_have_no_scientific_notation():
    table = options.defaults_table()
    assert [row[0] for row in table] == [p.name for p in PARAMS]
    assert not any("e-" in row[2] or "e+" in row[2] for row in table)
    assert ("min_order_qty", "0.002", "0.000001 – 1000", "quantité minimale d'un ordre de grille") in table
    assert dict((row[0], row[1]) for row in table)["spread"] == "0.015"


def test_no_source_file_is_silently_ignored_by_git():
    """`.gitignore` écarte des noms comme config.py (clés du bot d'origine) : un module portant ce nom n'aurait jamais été publié."""
    import shutil
    import subprocess

    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("pas de dépôt git ici")
    files = [str(p.relative_to(ROOT)) for pattern in ("src/**/*.py", "tests/**/*.py", "scripts/*.py", "examples/*") for p in ROOT.glob(pattern)]
    ignored = subprocess.run(["git", "check-ignore", *files], cwd=ROOT, capture_output=True, text=True)
    assert ignored.returncode == 1 and ignored.stdout == "", f"fichiers ignorés par git : {ignored.stdout}"
