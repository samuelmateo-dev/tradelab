"""Régénère les fichiers FICTIFS de data/synthetic/ (octet pour octet identiques à ceux du dépôt).

Usage :
    python scripts/make_synthetic.py [--out data/synthetic]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tradelab.synthetic import write_files


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=Path("data/synthetic"), type=Path)
    a = ap.parse_args()
    for path in write_files(a.out):
        print(f"{path} ({path.stat().st_size / 1e6:.2f} Mo)")


if __name__ == "__main__":
    main()
