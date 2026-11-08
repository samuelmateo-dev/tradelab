import numpy as np
import pandas as pd
import pytest

from tradelab.data import load_csv, load_klines_npz, load_month_dir, quality_report, resample_ohlcv

JUNE_2023_MINUTES = 28092960          # 2023-06-01 00:00 UTC, en MINUTES depuis 1970


def write_npz(path, start_min, n, price=100.0, step=1):
    ts = start_min + np.arange(n) * step
    c = price + np.arange(n, dtype=float)
    np.savez(path, ts=ts.astype("int64"), o=c - 0.5, h=c + 1.0, l=c - 1.0, c=c, v=np.ones(n), tb=np.ones(n) / 2)


def test_npz_timestamps_are_minutes_not_milliseconds(tmp_path):
    p = tmp_path / "2023-06.npz"
    write_npz(p, JUNE_2023_MINUTES, 5)
    df = load_klines_npz(p)
    assert df.index[0] == pd.Timestamp("2023-06-01 00:00:00", tz="UTC")
    assert df.index[1] - df.index[0] == pd.Timedelta(minutes=1)
    assert list(df.columns) == ["open", "high", "low", "close", "volume", "taker_buy"]


def test_pickled_object_arrays_are_refused(tmp_path):
    p = tmp_path / "evil.npz"
    np.savez(p, ts=np.arange(3), o=np.array([{"x": 1}, {"x": 2}, {"x": 3}], dtype=object))
    with pytest.raises(ValueError):
        load_klines_npz(p)


def test_file_without_timestamps_is_refused(tmp_path):
    p = tmp_path / "bad.npz"
    np.savez(p, o=np.ones(3))
    with pytest.raises(ValueError, match="ts"):
        load_klines_npz(p)


def test_month_directory_is_sorted_and_deduplicated_keeping_the_last(tmp_path):
    write_npz(tmp_path / "2023-07.npz", JUNE_2023_MINUTES + 10, 10, price=500.0)
    write_npz(tmp_path / "2023-06.npz", JUNE_2023_MINUTES, 15, price=100.0)       # recouvre 5 minutes
    df = load_month_dir(tmp_path)
    assert df.index.is_monotonic_increasing and not df.index.has_duplicates
    assert len(df) == 20
    assert df.loc[pd.Timestamp("2023-06-01 00:12", tz="UTC"), "close"] == 502.0   # le doublon prend le fichier trié en dernier


def test_empty_directory_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_month_dir(tmp_path)


def test_resample_aggregates_ohlcv_correctly(tmp_path):
    write_npz(tmp_path / "a.npz", JUNE_2023_MINUTES, 120)
    df = load_klines_npz(tmp_path / "a.npz")
    h = resample_ohlcv(df, "1h")
    assert len(h) == 2
    first = h.iloc[0]
    assert first["open"] == df["open"].iloc[0] and first["close"] == df["close"].iloc[59]
    assert first["high"] == df["high"].iloc[:60].max() and first["low"] == df["low"].iloc[:60].min()
    assert first["volume"] == 60


def test_resample_drops_empty_periods_instead_of_inventing_them(tmp_path):
    write_npz(tmp_path / "a.npz", JUNE_2023_MINUTES, 60)
    write_npz(tmp_path / "b.npz", JUNE_2023_MINUTES + 180, 60, price=200.0)      # trou de 2 heures
    h = resample_ohlcv(load_month_dir(tmp_path), "1h")
    assert len(h) == 2


def hourly(n=10):
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    c = np.linspace(100, 110, n)
    return pd.DataFrame({"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 1.0}, index=idx)


def test_quality_report_on_clean_data():
    q = quality_report(hourly(), "1h")
    assert q.rows == 10 and q.missing_rows == 0 and q.clean and q.missing_share == 0.0


def test_quality_report_counts_missing_hours():
    df = hourly().drop(hourly().index[[3, 4]])
    q = quality_report(df, "1h")
    assert q.missing_rows == 2 and q.expected_rows == 10 and q.missing_share == pytest.approx(0.2)


def test_quality_report_flags_inconsistent_and_invalid_candles():
    df = hourly()
    df.iloc[2, df.columns.get_loc("high")] = df["close"].iloc[2] - 5      # haut sous la clôture
    df.iloc[5, df.columns.get_loc("close")] = -1.0
    q = quality_report(df, "1h")
    assert q.inconsistent_ohlc >= 1 and q.non_positive_prices == 1 and not q.clean


def test_quality_report_rejects_empty_data():
    with pytest.raises(ValueError):
        quality_report(hourly().iloc[:0])


def test_csv_loader_returns_a_utc_sorted_index(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("time,open,high,low,close,volume\n2024-01-01 01:00:00,2,3,1,2,1\n2024-01-01 00:00:00,1,2,0,1,1\n", encoding="utf-8")
    df = load_csv(p)
    assert str(df.index.tz) == "UTC" and df.index.is_monotonic_increasing
