#!/usr/bin/env python3
"""
Build the merged MADAR-26 table (``data/MADAR/MADAR.tsv``) from the per-city
corpus files shipped under ``data/MADAR/MADAR.Parallel-Corpora-...``.

Takes Beirut as the base frame (it carries the ``split`` column), left-merges
every other city on ``sentID.BTEC``, renames each ``sent`` column to its short
dialect code, indexes by ``sentID.BTEC`` and writes TSV. ``data/MADAR/MADAR.tsv``
ships with the repo; run this to regenerate it.

Usage (from the repo root):
  python -m MADAR_alignment.build_madar_table
  python -m MADAR_alignment.build_madar_table --corpus-dir <dir> --out <path>
"""

import argparse
import logging
import os

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("build_madar_table")

DEFAULT_CORPUS_DIR = os.path.join(
    "data", "MADAR", "MADAR.Parallel-Corpora-Public-Version1.1-25MAR2021", "MADAR_Corpus"
)
DEFAULT_OUT = os.path.join("data", "MADAR", "MADAR.tsv")

# (city file stem, short dialect code) in the canonical MADAR.tsv column order.
# Beirut is handled separately as the base frame (it keeps the `split` column).
CITY_CODES = [
    ("Alexandria", "ALEX"), ("Amman", "AMM"), ("Aswan", "ASW"), ("Aleppo", "ALE"),
    ("Cairo", "CAI"), ("Damascus", "DAM"), ("Jerusalem", "JER"), ("Salt", "SAL"),
    ("MSA", "MSA"), ("Doha", "DOH"), ("Rabat", "RAB"), ("Tunis", "TUN"),
    ("Algiers", "ALG"), ("Baghdad", "BAG"), ("Basra", "BAS"), ("Benghazi", "BEN"),
    ("Fes", "FES"), ("Jeddah", "JED"), ("Khartoum", "KHA"), ("Mosul", "MOS"),
    ("Muscat", "MUS"), ("Riyadh", "ARI"), ("Sanaa", "SAN"), ("Sfax", "SFX"),
    ("Tripoli", "TRI"),
]


def _read_city(corpus_dir: str, stem: str) -> pd.DataFrame:
    path = os.path.join(corpus_dir, f"MADAR.corpus.{stem}.tsv")
    return pd.read_csv(path, delimiter="\t")


def build_madar_table(corpus_dir: str = DEFAULT_CORPUS_DIR, out_path: str = DEFAULT_OUT) -> pd.DataFrame:
    logger.info("Reading per-city corpora from: %s", corpus_dir)
    beirut = _read_city(corpus_dir, "Beirut")
    madar = beirut[["sentID.BTEC", "split", "sent"]].rename(columns={"sent": "BEI"})

    for stem, code in CITY_CODES:
        city = _read_city(corpus_dir, stem)[["sentID.BTEC", "sent"]].rename(columns={"sent": code})
        madar = madar.merge(city, on="sentID.BTEC", how="left")

    madar.set_index("sentID.BTEC", inplace=True)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    madar.to_csv(out_path, sep="\t")
    logger.info("Wrote %d rows x %d columns -> %s", len(madar), madar.shape[1], out_path)
    return madar


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus-dir", default=DEFAULT_CORPUS_DIR, help="Directory containing MADAR.corpus.<City>.tsv files")
    p.add_argument("--out", default=DEFAULT_OUT, help="Output TSV path")
    return p.parse_args()


def main(args):
    build_madar_table(args.corpus_dir, args.out)


if __name__ == "__main__":
    main(parse_args())
