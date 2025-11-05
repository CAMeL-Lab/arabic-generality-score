#!/usr/bin/env python3
"""
Run awesome-align on (MSA ||| DIALECT) pairs using a finetuned checkpoint.

Equivalent to:
  !awesome-align --data_file=... --output_file=... --model_name_or_path=... --batch_size=32
"""

import os
import sys
import argparse
import logging
import subprocess
import pandas as pd
from copy import deepcopy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utilities.preprocess_text import preprocess_text  # noqa: E402
# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("run_alignment")
dialects_26 = [
    "BEI", "ALEX", "AMM", "ASW", "ALE", "CAI", "DAM", "JER", "SAL",
    "MSA", "DOH", "RAB", "TUN", "ALG", "BAG", "BAS", "BEN", "FES",
    "JED", "KHA", "MOS", "MUS", "ARI", "SAN", "SFX", "TRI"
]
def import_MADAR():
    """
    Load MADAR TSV and preprocess only the dialect columns with your `preprocess_text`.
    Keeps NaNs as-is. Fails early if expected columns are missing.
    """
    path = "data/MADAR/MADAR.tsv"
    logger.info(f"Loading MADAR from: {path}")
    MADAR_df = pd.read_csv(path, sep="\t", encoding="utf-8")

    # Sanity-check expected columns
    missing = [c for c in dialects_26 if c not in MADAR_df.columns]
    if missing:
        raise ValueError(
            f"Missing MADAR columns: {missing}. "
            f"Present sample: {sorted(MADAR_df.columns.tolist())[:12]}..."
        )

    logger.info("Preprocessing dialect columns with `preprocess_text` (NaNs preserved)")
    # Apply preprocess_text per cell on the selected columns
    MADAR_df[dialects_26] = MADAR_df[dialects_26].map(
        lambda x: preprocess_text(x) if pd.notnull(x) else x
    )

    logger.info(f"MADAR loaded. Rows: {len(MADAR_df)}")
    return MADAR_df



def parse_args():
    p = argparse.ArgumentParser(description="Run awesome-align with a finetuned model checkpoint.")
    p.add_argument("--data_file", required=True, help="Path to parallel data file: lines like 'SRC ||| TGT'.")
    p.add_argument("--output_file", required=True, help="Where to write alignments (one line per input pair).")
    p.add_argument("--model_name_or_path", required=True, help="Finetuned checkpoint dir or HF model ID.")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size for alignment.")
    p.add_argument("--extraction", default="softmax", choices=["softmax", "softmax_threshold", "softmax_saliency", "softmax_sparsemax"],
                   help="Extraction method (awesome-align flag). Default: softmax")
    return p.parse_args()



def run_alignment():
    args = parse_args()

    # Sanity checks
    if not os.path.isfile(args.data_file):
        raise FileNotFoundError(f"data_file not found: {args.data_file}")
    if not (os.path.isdir(args.model_name_or_path) or isinstance(args.model_name_or_path, str)):
        raise FileNotFoundError(f"model_name_or_path not found: {args.model_name_or_path}")
    out_dir = os.path.dirname(os.path.abspath(args.output_file)) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Build command (module form is robust across environments)
    cmd = [
        sys.executable, "-m", "awesome_align.run_align",
        "--data_file", args.data_file,
        "--output_file", args.output_file,
        "--model_name_or_path", args.model_name_or_path,
        "--batch_size", str(args.batch_size),
        "--extraction", args.extraction,
        "--num_workers", "0"
    ]

    logger.info("Launching awesome-align:\n%s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        logger.info("✅ Alignments written to: %s", args.output_file)
    except FileNotFoundError:
        logger.error("awesome-align is not installed in this environment. Try: pip install awesome-align")
        raise
    except subprocess.CalledProcessError as e:
        logger.exception("awesome-align failed with exit code %s", e.returncode)
        raise



if __name__ == "__main__":
    run_alignment()

