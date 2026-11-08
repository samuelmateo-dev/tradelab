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
