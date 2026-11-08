import gzip

import numpy as np
import pandas as pd
import pytest

from tradelab.data import load_csv, quality_report
from tradelab.synthetic import BARS_FILE, FUNDING_FILE, synthetic_bars, synthetic_funding, write_files

SHORT = dict(start="2020-01-01 00:00", end="2020-03-31 23:00")


def test_same_seed_same_bars_and_different_seed_differs():
    a, b, c = synthetic_bars(**SHORT, seed=1), synthetic_bars(**SHORT, seed=1), synthetic_bars(**SHORT, seed=2)
    pd.testing.assert_frame_equal(a, b)
    assert not a["close"].equals(c["close"])


def test_bars_are_complete_consistent_and_hourly():
    df = synthetic_bars()
    q = quality_report(df, "1h")
    assert q.missing_rows == 0 and q.clean
    assert df.index[0].year == 2020 and df.index[-1].year == 2026
    assert (df.index.to_series().diff().dropna() == pd.Timedelta("1h")).all()
    assert (df["volume"] > 0).all()


def test_bars_have_the_requested_annual_volatility_and_fat_tails():
    df = synthetic_bars(annual_vol=0.75)
    r = np.log(df["close"]).diff().dropna()
    assert r.std() * np.sqrt(8760) == pytest.approx(0.75, rel=0.1)
    assert r.kurt() > 3        # queues plus épaisses qu'une loi normale


def test_each_bar_opens_where_the_previous_one_closed():
    df = synthetic_bars(**SHORT)
    assert (df["open"].iloc[1:].to_numpy() == df["close"].iloc[:-1].to_numpy()).all()


def test_funding_is_daily_covers_every_year_and_has_negative_days():
    s = synthetic_funding()
    assert s.index[0].year == 2020 and s.index[-1].year == 2026
    assert (s.index.to_series().diff().dropna() == pd.Timedelta("1D")).all()
    assert (s < 0).any() and (s > 0).any()


def test_write_files_is_byte_identical_between_runs(tmp_path):
    a = write_files(tmp_path / "a")
    b = write_files(tmp_path / "b")
    assert [p.name for p in a] == [BARS_FILE, FUNDING_FILE]
    for pa, pb in zip(a, b):
        assert pa.read_bytes() == pb.read_bytes()


def test_files_committed_in_the_repo_match_the_generator(tmp_path):
    """Si le générateur change, il faut régénérer data/synthetic/ (python scripts/make_synthetic.py)."""
    from pathlib import Path

    repo = Path(__file__).resolve().parent.parent / "data" / "synthetic"
    fresh = write_files(tmp_path)
    for p in fresh:
        committed = repo / p.name
        assert committed.exists(), f"{committed} manquant : lancer scripts/make_synthetic.py"
        opener = gzip.open if p.suffix == ".gz" else open
        with opener(p, "rb") as x, opener(committed, "rb") as y:
            assert x.read() == y.read(), f"{committed.name} ne correspond plus au générateur"


def test_load_csv_round_trips_the_written_bars(tmp_path):
    paths = write_files(tmp_path)
    loaded = load_csv(paths[0])
    assert quality_report(loaded, "1h").clean
    assert len(loaded) == len(synthetic_bars())
