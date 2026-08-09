#!/usr/bin/env python3
"""
Reference baselines for the MDID sentence-generality task
(``Evaluation-Inference.ipynb`` cells 47-62 and ``baseline_word_lookup.ipynb``).

  constant : predict the dev-set mean generality for every test sentence
  random   : predict Uniform(0, 1)
  topp     : #dialects from AMR-KELEG/NADI2024-baseline top-p (P=0.9), / 18
  b2bert   : #dialects from AHAAM/B2BERT (sigmoid > 0.3), / 18
  lexicon  : max AGS of any word in the sentence, looked up in the AGS table, averaged

``topp``/``b2bert`` download models from the Hub; ``lexicon`` needs --agg-table.
The MDID CSV is not bundled (see data/README.md).

Usage (from the repo root):
  python -m AGS_evaluation.baselines --mdid-csv path/to/NADI2024_full_extended.csv --baseline all
"""

import argparse
import logging

import numpy as np

from AGS_evaluation.evaluate_mdid import _rmse, load_mdid
from utilities.preprocess_text import preprocess_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("baselines")

DIALECTS_18 = [
    "Algeria", "Bahrain", "Egypt", "Iraq", "Jordan", "Kuwait", "Lebanon", "Libya",
    "Morocco", "Oman", "Palestine", "Qatar", "Saudi_Arabia", "Sudan", "Syria",
    "Tunisia", "UAE", "Yemen",
]


def baseline_constant(mdid):
    dev_mean = mdid[mdid["split"] == "dev"]["generality"].mean()
    test = mdid[mdid["split"] == "test"]
    return _rmse(test["generality"], [dev_mean] * len(test))


def baseline_random(mdid, seed: int = 0):
    test = mdid[mdid["split"] == "test"]
    rng = np.random.default_rng(seed)
    return _rmse(test["generality"], rng.uniform(0, 1, size=len(test)))


def baseline_topp(mdid, p: float = 0.9):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("AMR-KELEG/NADI2024-baseline")
    model = AutoModelForSequenceClassification.from_pretrained("AMR-KELEG/NADI2024-baseline")

    def predict_top_p(text):
        logits = model(**tok(text, return_tensors="pt")).logits
        probs = torch.softmax(logits, dim=1).flatten().tolist()
        order = torch.topk(logits, 18).indices.flatten().tolist()
        picked, total = 0, 0.0
        for i in order:
            total += probs[i]
            picked += 1
            if total >= p:
                break
        return picked

    test = mdid[mdid["split"] == "test"]
    preds = [predict_top_p(s) / 18 for s in test["sentence"]]
    return _rmse(test["generality"], preds)


def baseline_b2bert(mdid, threshold: float = 0.3):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("AHAAM/B2BERT")
    model = AutoModelForSequenceClassification.from_pretrained("AHAAM/B2BERT")

    def n_valid(text):
        enc = tok([text], truncation=True, padding=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        return int((probs >= threshold).sum())

    test = mdid[mdid["split"] == "test"]
    preds = [n_valid(s) / 18 for s in test["sentence"]]
    return _rmse(test["generality"], preds)


def baseline_lexicon(mdid, agg_table: str, score_col: str = "min_t_0.5_k_20_c6"):
    from AGS_training.build_training_data import load_agg_table

    df = load_agg_table(agg_table)
    lookup = {}
    for _, row in df.iterrows():
        col = row["dialect"]
        mapping = row.get(col)
        if isinstance(mapping, dict):
            for w in mapping:
                if w:
                    lookup[w] = max(lookup.get(w, 0.0), float(row[score_col]))

    def sentence_score(sentence):
        scores = []
        for w in preprocess_text(sentence).split():
            if w in lookup:
                scores.append(lookup[w])
            else:
                norm = preprocess_text(w, normalize=True)
                scores.append(lookup.get(norm, -1))
        return sum(scores) / len(scores) if scores else -1

    test = mdid[mdid["split"] == "test"]
    preds = [sentence_score(s) for s in test["sentence"]]
    return _rmse(test["generality"], preds)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mdid-csv", required=True)
    p.add_argument("--baseline", choices=["constant", "random", "topp", "b2bert", "lexicon", "all"], default="all")
    p.add_argument("--agg-table", default="output/MADAR_26_word_alignment_aug_agg.tsv",
                   help="AGS table for the lexicon baseline")
    p.add_argument("--score-col", default="min_t_0.5_k_20_c6")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main(args):
    mdid = load_mdid(args.mdid_csv)
    todo = ["constant", "random", "topp", "b2bert", "lexicon"] if args.baseline == "all" else [args.baseline]
    for name in todo:
        try:
            if name == "constant":
                rmse = baseline_constant(mdid)
            elif name == "random":
                rmse = baseline_random(mdid, args.seed)
            elif name == "topp":
                rmse = baseline_topp(mdid)
            elif name == "b2bert":
                rmse = baseline_b2bert(mdid)
            else:
                rmse = baseline_lexicon(mdid, args.agg_table, args.score_col)
            print(f"{name:9s} | test RMSE = {rmse:.4f}")
        except Exception as e:  # noqa: BLE001 — one failing baseline shouldn't kill the rest
            logger.warning("baseline %s failed: %s", name, e)


if __name__ == "__main__":
    main(parse_args())
