#!/usr/bin/env python3
"""
Evaluate an AGS checkpoint on MDID (``Evaluation-Inference.ipynb`` cells 11-19).

MDID = the NADI-2024 subtask-1 sentences with per-country validity annotations.
The gold sentence-level generality is ``n_valid / 11``; the prediction is the
harmonic mean of the ``k`` lowest per-word AGS scores.

The input CSV (``NADI2024_full_extended.csv``) is NOT bundled — it is derived from
the NADI-2024 shared-task data (licence-restricted). See data/README.md.

Usage (from the repo root):
  python -m AGS_evaluation.evaluate_mdid \\
    --mdid-csv path/to/NADI2024_full_extended.csv \\
    --tokenizer-dir models/ags_tokenizer \\
    --checkpoint models/ags_sentence_madar6/checkpoint_step5500.pt \\
    --split both --k 2
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd

from AGS_evaluation._model import (
    load_gen_model, load_tokenizer, predict_score, predict_scores_sent, sentence_generality,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("evaluate_mdid")


def _rmse(gold, pred) -> float:
    return float(((np.asarray(gold, float) - np.asarray(pred, float)) ** 2).mean() ** 0.5)


def load_mdid(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"MDID CSV not found: {path}\n"
            "This file is not bundled (NADI-2024 licence). See data/README.md."
        )
    df = pd.read_csv(path)
    df["generality"] = df["n_valid"].map(lambda x: x / 11)
    return df


def evaluate(df: pd.DataFrame, model, tokenizer, k: int, whole: bool) -> dict:
    if whole:
        preds = [predict_score(model, tokenizer, s, s) for s in df["sentence"]]
    else:
        preds = [sentence_generality(predict_scores_sent(model, tokenizer, s), k) for s in df["sentence"]]
    return {"rmse": _rmse(df["generality"], preds), "n": len(df)}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mdid-csv", required=True)
    p.add_argument("--tokenizer-dir", default="models/ags_tokenizer")
    p.add_argument("--checkpoint", required=True, nargs="+", help="One or more .pt checkpoints")
    p.add_argument("--split", choices=["dev", "test", "both"], default="both")
    p.add_argument("--k", type=int, default=2, help="Use the k lowest word scores")
    p.add_argument("--whole", action="store_true", help="Score the whole sentence as its own target")
    return p.parse_args()


def main(args):
    mdid = load_mdid(args.mdid_csv)
    tokenizer = load_tokenizer(args.tokenizer_dir)
    splits = ["dev", "test"] if args.split == "both" else [args.split]

    for ckpt in args.checkpoint:
        logger.info("Loading %s", ckpt)
        model = load_gen_model(ckpt, tokenizer)
        for split in splits:
            sub = mdid[mdid["split"] == split]
            res = evaluate(sub, model, tokenizer, args.k, args.whole)
            print(f"{os.path.basename(ckpt):45s} | {split:4s} | n={res['n']:5d} | RMSE={res['rmse']:.4f}")


if __name__ == "__main__":
    main(parse_args())
