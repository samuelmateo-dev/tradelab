import json
from pathlib import Path

import pytest

from tradelab.cli import build_parser, fr, main, pct, table
from tradelab.data import load_csv

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_FUNDING = ROOT / "data" / "synthetic" / "btc_funding.csv"
SAMPLE_BARS = ROOT / "data" / "synthetic" / "eth_1h.csv.gz"


def test_french_number_formatting():
    assert fr(12.345, 1) == "12,3" and fr(1.5, 1, " %") == "1,5 %" and fr(float("nan")) == "—"
    assert pct(0.1234) == "12,3 %"


def test_table_aligns_columns():
    out = table(["A", "Valeur"], [["x", "1"], ["long", "22"]]).splitlines()
    assert len({len(line) for line in out}) == 1


def test_missing_files_return_an_error_code(capsys, tmp_path):
    assert main(["carry", "--funding", str(tmp_path / "nope.csv")]) == 1
    assert main(["backtest", "--data", str(tmp_path / "nope.csv")]) == 1
    assert "introuvable" in capsys.readouterr().err


def test_carry_command_reports_every_year_of_the_synthetic_file(capsys):
    assert main(["carry", "--funding", str(SAMPLE_FUNDING)]) == 0
    out = capsys.readouterr().out
    for year in range(2020, 2027):
        assert str(year) in out
    assert "12 derniers mois" in out and "Hors mesure" in out


def test_report_says_when_the_data_is_fictitious(capsys):
    assert main(["carry", "--funding", str(SAMPLE_FUNDING)]) == 0
    assert "FICTIVES" in capsys.readouterr().out


def test_no_fictitious_warning_for_other_files(capsys, tmp_path):
    df = load_csv(SAMPLE_FUNDING)
    p = tmp_path / "mon_financement.csv"
    df.to_csv(p, index_label="date")
    assert main(["carry", "--funding", str(p)]) == 0
    assert "FICTIVES" not in capsys.readouterr().out


def test_backtest_json_on_a_three_year_slice(capsys, tmp_path):
    df = load_csv(SAMPLE_BARS).loc["2021-01-01":"2023-12-31"]
    p = tmp_path / "slice.csv"
    df.to_csv(p, index_label="time")
    assert main(["backtest", "--data", str(p), "--split", "2023-01-01", "--n-boot", "200", "--json"]) == 0
    verdicts = json.loads(capsys.readouterr().out)
    assert len(verdicts) == 2
    assert all(v["demonstrated"] is False for v in verdicts)            # 12 mois de validation : jamais assez
    assert all(v["alpha_adjusted"] == pytest.approx(0.025) for v in verdicts)


def test_backtest_text_report_has_the_expected_sections(capsys, tmp_path):
    df = load_csv(SAMPLE_BARS).loc["2021-01-01":"2023-12-31"]
    p = tmp_path / "slice.csv"
    df.to_csv(p, index_label="time")
    assert main(["backtest", "--data", str(p), "--split", "2023-01-01", "--n-boot", "200"]) == 0
    out = capsys.readouterr().out
    assert "VERDICT" in out and "Bonferroni" in out and "Bot tel quel" in out and "sans levier" in out


def test_parser_requires_a_subcommand():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


# --- Choix de l'utilisateur : paramètres, leviers, analyses (voir aussi test_options.py) ---------------

@pytest.fixture(scope="module")
def slice_csv(tmp_path_factory):
    """Dix-huit mois de bougies fictives (6 de contexte, 12 de validation) : le juge tourne, et les tests restent rapides."""
    path = tmp_path_factory.mktemp("slice") / "slice.csv"
    load_csv(SAMPLE_BARS).loc["2022-07-01":"2023-12-31"].to_csv(path, index_label="time")
    return path


def _backtest(slice_csv, capsys, *extra, json_out=True):
    argv = ["backtest", "--data", str(slice_csv), "--split", "2023-01-01", "--n-boot", "200", *(["--json"] if json_out else []), *extra]
    code = main(argv)
    out = capsys.readouterr().out
    return code, (json.loads(out) if json_out and code == 0 else out)


def test_params_command_lists_everything_that_can_be_chosen(capsys):
    from tradelab import options

    assert main(["params"]) == 0
    out = capsys.readouterr().out
    for name, default, span, _ in options.defaults_table():
        assert name in out and span in out
    assert "adaptive" in out and "×5 dès 500 de capital" in out and "×2 sinon" in out
    for analysis in options.ANALYSES:
        assert analysis in out
    assert options.EXAMPLE_TEXT.rstrip() in out and "--tests-counted" in out and "--leverage" in out
    assert "Reste figé" in out and "24 mois" in out


def test_default_report_is_unchanged_by_the_new_options(slice_csv, capsys):
    code, out = _backtest(slice_csv, capsys, json_out=False)
    assert code == 0
    assert "Graine 42 · 200 tirages · α corrigé (Bonferroni, 2 hypothèses) = 0,025\n" in out
    assert "Paramètres de la stratégie modifiés" not in out and "Stratégie modifiée" not in out and "tests déclarés" not in out
    assert "▌ Bot tel quel (levier adaptatif ×2/×3/×5)\n" in out and "▌ Même stratégie sans levier (×1)\n" in out
    assert "Sensibilité" not in out


def test_set_changes_the_strategy_and_the_report_says_so(slice_csv, capsys):
    _, default = _backtest(slice_csv, capsys)
    code, modified = _backtest(slice_csv, capsys, "--set", "levels=4", "--set", "spread=0,03")
    assert code == 0
    assert [v["hypothesis"] for v in modified] == [
        "Stratégie modifiée (levier adaptatif ×2/×3/×5) — levels=4, spread=0.03",
        "Même stratégie sans levier (×1) — levels=4, spread=0.03",
    ]
    # les paramètres arrivent bien jusqu'à la stratégie : les résultats changent
    assert [v["val"]["total_return"] for v in modified] != [v["val"]["total_return"] for v in default]
    code, text = _backtest(slice_csv, capsys, "--set", "levels=4", json_out=False)
    assert "Paramètres de la stratégie modifiés : levels=4 (les autres : valeurs du bot d'origine)" in text
    assert "AVANT de regarder la fenêtre VAL" in text and "--tests-counted" in text


def test_params_file_is_read_and_set_wins_over_it(slice_csv, capsys, tmp_path):
    file = tmp_path / "p.json"
    file.write_text(json.dumps({"spread": 0.02, "levels": 6}), encoding="utf-8")
    _, from_file = _backtest(slice_csv, capsys, "--params", str(file), "--leverage", "1")
    assert from_file[0]["hypothesis"] == "Même stratégie sans levier (×1) — levels=6, spread=0.02"
    _, overridden = _backtest(slice_csv, capsys, "--params", str(file), "--set", "spread=0.03", "--leverage", "1")
    assert overridden[0]["hypothesis"] == "Même stratégie sans levier (×1) — levels=6, spread=0.03"
    _, via_set = _backtest(slice_csv, capsys, "--set", "levels=6", "--set", "spread=0.03", "--leverage", "1")
    assert overridden == via_set  # --params + --set équivaut à tout donner avec --set


def test_leverage_list_makes_one_hypothesis_per_level_and_bonferroni_counts_them(slice_csv, capsys):
    code, verdicts = _backtest(slice_csv, capsys, "--leverage", "1,2,3")
    assert code == 0 and len(verdicts) == 3
    assert [v["hypothesis"] for v in verdicts] == ["Même stratégie sans levier (×1)", "Même stratégie, levier fixe ×2", "Même stratégie, levier fixe ×3"]
    assert all(v["alpha_adjusted"] == pytest.approx(0.05 / 3) for v in verdicts)
    one = _backtest(slice_csv, capsys, "--leverage", "1")[1][0]
    assert one["alpha_adjusted"] == pytest.approx(0.05)


def test_the_same_hypothesis_gives_the_same_numbers_whatever_it_is_run_with(slice_csv, capsys):
    """Le résultat de la stratégie ×1 ne dépend pas des autres hypothèses lancées (seul α corrigé change)."""
    both = _backtest(slice_csv, capsys)[1][1]
    alone = _backtest(slice_csv, capsys, "--leverage", "1")[1][0]
    assert both["hypothesis"] == alone["hypothesis"]
    assert both["val"]["total_return"] == alone["val"]["total_return"] and both["dev"]["total_return"] == alone["dev"]["total_return"]
    assert both["alpha_adjusted"] == pytest.approx(0.025) and alone["alpha_adjusted"] == pytest.approx(0.05)


def test_alpha_and_tests_counted_feed_the_bonferroni_correction(slice_csv, capsys):
    assert _backtest(slice_csv, capsys, "--alpha", "0.1")[1][0]["alpha_adjusted"] == pytest.approx(0.05)
    assert _backtest(slice_csv, capsys, "--tests-counted", "10")[1][0]["alpha_adjusted"] == pytest.approx(0.005)
    _, text = _backtest(slice_csv, capsys, "--tests-counted", "10", "--alpha", "0.1", json_out=False)
    assert "α de départ 0,100 · α corrigé (Bonferroni, 10 tests déclarés, dont 2 lancés ici) = 0,010" in text


def test_analyses_can_be_chosen(slice_csv, capsys):
    _, sensitivity_only = _backtest(slice_csv, capsys, "--analyses", "sensitivity", json_out=False)
    assert "Sensibilité aux coûts" in sensitivity_only and "aucun coût" in sensitivity_only
    assert "VERDICT" not in sensitivity_only and "Graine" not in sensitivity_only and "Bonferroni" not in sensitivity_only
    _, both = _backtest(slice_csv, capsys, "--analyses", "verdict,sensitivity", json_out=False)
    _, legacy = _backtest(slice_csv, capsys, "--sensitivity", json_out=False)
    assert both == legacy and "VERDICT" in both and "Sensibilité aux coûts" in both


def test_sensitivity_uses_the_chosen_parameters(slice_csv, capsys):
    _, default = _backtest(slice_csv, capsys, "--analyses", "sensitivity", json_out=False)
    _, modified = _backtest(slice_csv, capsys, "--analyses", "sensitivity", "--set", "levels=4", json_out=False)
    assert "▌ Sensibilité aux coûts (période complète, sans levier, capital neuf)\n" in default
    assert "▌ Sensibilité aux coûts (période complète, sans levier, capital neuf) — levels=4\n" in modified
    assert default.split("Scénario")[1] != modified.split("Scénario")[1]


@pytest.mark.parametrize("args, message", [
    (["--set", "bogus=1"], "bogus : paramètre inconnu"),
    (["--set", "spread=5"], "spread : 5.0 hors de la plage acceptée [0.001 ; 0.2]"),
    (["--set", "levels=2.5"], "levels : entier attendu"),
    (["--set", "spread"], "attendu nom=valeur"),
    (["--set", "spread=0.02", "--set", "spread=0.03"], "défini deux fois"),
    (["--set", "fixed_leverage=2"], "le levier se choisit avec --leverage"),
    (["--set", "rsi_cover=70"], "rsi_cover (70) doit être inférieur à rsi_short (65)"),
    (["--params", "absent.json"], "--params : fichier introuvable"),
    (["--params", "mal.json"], "JSON invalide"),
    (["--params", "faute.json"], "spred : paramètre inconnu"),
    (["--leverage", "0"], "×0 hors de la plage"),
    (["--leverage", "adaptive,adaptive"], "donné deux fois"),
    (["--leverage", "x"], "« x » invalide"),
    (["--analyses", "foo"], "« foo » inconnue"),
    (["--tests-counted", "1"], "--tests-counted 1 : au moins 2"),
    (["--alpha", "0"], "seuil entre 0 et 1"),
    (["--alpha", "1.5"], "seuil entre 0 et 1"),
    (["--analyses", "sensitivity", "--leverage", "2"], "--leverage n'a d'effet qu'avec l'analyse « verdict »"),
    (["--analyses", "sensitivity", "--alpha", "0.1"], "--alpha n'a d'effet qu'avec l'analyse « verdict »"),
    (["--analyses", "sensitivity", "--tests-counted", "3"], "--tests-counted n'a d'effet qu'avec l'analyse « verdict »"),
    (["--analyses", "sensitivity", "--json"], "--json n'a d'effet qu'avec l'analyse « verdict »"),
    (["--json", "--sensitivity"], "--json ne sort que les verdicts"),
])
def test_invalid_options_stop_before_any_data_is_read(tmp_path, capsys, args, message):
    (tmp_path / "mal.json").write_text("{ pas du json", encoding="utf-8")
    (tmp_path / "faute.json").write_text(json.dumps({"spred": 0.02}), encoding="utf-8")
    args = [str(tmp_path / a) if a.endswith(".json") else a for a in args]
    # --data n'existe pas : si l'erreur affichée est celle de l'option, c'est que la validation passe avant la lecture des données
    assert main(["backtest", "--data", str(tmp_path / "nope.csv"), *args]) == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("Erreur : ") and message in captured.err and "introuvable : " + str(tmp_path / "nope.csv") not in captured.err
    assert captured.out == ""


def test_valid_options_with_a_missing_data_file_still_report_the_missing_file(tmp_path, capsys):
    assert main(["backtest", "--data", str(tmp_path / "nope.csv"), "--set", "levels=6", "--leverage", "1,2"]) == 1
    assert "Fichier introuvable" in capsys.readouterr().err
