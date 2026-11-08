"""Convertit les archives .npz de l'auteur en CSV réels dans data/real/ (dossier ignoré par git).

Les fichiers produits viennent de données Binance : ne pas les ajouter au dépôt (voir README).

Usage :
    python scripts/build_sample.py --eth-dir <dossier ETH/ avec 2020-01.npz …> \
                                   --btc-daily <BTCUSDT.npz> --btc-funding <fund_BTCUSDT.npz> \
                                   --out data/real

Les .npz d'origine : clés ts (minutes depuis 1970), o, h, l, c, v, tb pour les bougies d'une minute ;
d (jours depuis 1970), o, h, l, c, v … pour le journalier ; d et v pour le financement.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tradelab.data import load_month_dir, resample_ohlcv


def _days(d: np.ndarray) -> pd.DatetimeIndex:
    return pd.to_datetime(d.astype("int64") * 86400, unit="s", utc=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eth-dir", required=True, type=Path)
    ap.add_argument("--btc-daily", required=True, type=Path)
    ap.add_argument("--btc-funding", required=True, type=Path)
    ap.add_argument("--out", default=Path("data/real"), type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    one_min = load_month_dir(a.eth_dir)
    hourly = resample_ohlcv(one_min, "1h")
    hourly.to_csv(a.out / "eth_1h.csv.gz", index_label="time", float_format="%.6g")
    print(f"eth_1h.csv.gz : {len(hourly)} bougies, {hourly.index[0]} → {hourly.index[-1]}")

    z = np.load(a.btc_daily, allow_pickle=False)
    daily = pd.DataFrame(
        {k: z[k] for k in ("o", "h", "l", "c", "v")}, index=_days(z["d"])
    ).rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    daily.to_csv(a.out / "btc_1d.csv", index_label="date", float_format="%.8g")
    print(f"btc_1d.csv : {len(daily)} jours")

    f = np.load(a.btc_funding, allow_pickle=False)
    funding = pd.Series(f["v"], index=_days(f["d"]), name="funding_sum")
    funding.to_csv(a.out / "btc_funding.csv", index_label="date", float_format="%.8g")
    print(f"btc_funding.csv : {len(funding)} jours")


if __name__ == "__main__":
    main()
