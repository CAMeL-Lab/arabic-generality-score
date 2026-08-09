#!/usr/bin/env python3
# AGS_extraction/AGS_extraction.py
# Compute word-level AGS from MADAR reformatted alignments using augmented distances.

import os
import ast
import math
import numpy as np
import pandas as pd

# substitution weights & distance calculator
from distance_function.substitution_weight import (
    load_distance_resources,
    compute_substitution_cost,
)
from distance_function.augmented_edit_distance import DistanceCalculator  # uses same substitution function

# ----------------- constants -----------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.environ.get("AGS_OUTPUT_DIR", os.path.join(ROOT, "output"))
ALIGN_PATH = os.path.join(OUTPUT_DIR, "MADAR_reformatted_word_alignments.tsv")
SCORES_OUT = os.path.join(OUTPUT_DIR, "AGS_scores.tsv")
# Full table with the dict-valued distance/score columns kept; consumed by
# AGS_training/build_training_data.py.
FULL_OUT = os.path.join(OUTPUT_DIR, "MADAR_26_word_alignment_aug_agg.tsv")

dialects_26 = [
    "MSA", "BEI", "ALEX", "AMM", "ASW", "ALE", "CAI",
    "DAM", "JER", "SAL", "DOH", "RAB", "TUN", "ALG",
    "BAG", "BAS", "BEN", "FES", "JED", "KHA", "MOS",
    "MUS", "ARI", "SAN", "SFX", "TRI"
]
dialects_6 = ["BEI", "CAI", "MSA", "DOH", "RAB", "TUN"]

# ----------------- helpers -----------------
def determine_corpus(row: pd.Series, extra_dialects: set) -> int:
    """Return 26 if any extra dialect col has a non-empty dict; else 6."""
    for dialect in extra_dialects:
        val = row.get(dialect)
        if isinstance(val, dict) and set(val.keys()) != {None} and len(val) > 0:
            return 26
    return 6

def freq_inverse_softmax_aggregate(distances, freqs, temperature: float = 5.0):
    """
    Aggregate distances with weights ∝ freq * softmax(-temperature * d).
    Smaller distances => larger weight. Returns a single aggregated distance.
    """
    d = np.array(distances, dtype=float)
    f = np.array(freqs, dtype=float)
    if d.size == 0 or f.sum() == 0:
        return None
    logits = -temperature * d
    # numerically stable softmax
    m = np.max(logits)
    w = np.exp(logits - m)
    w = w / w.sum()
    w = w * f
    if w.sum() == 0:
        return float(d.mean())
    w = w / w.sum()
    return float((w * d).sum())

def smooth_threshold(d, t=0.3, k=10.0) -> float:
    """
    Logistic gate ~1 for d << t, ~0 for d >> t:
      f(d) = 1 / (1 + exp(-k * (t - d)))
    """
    return 1.0 / (1.0 + math.exp(-k * (t - d)))

def smooth_threshold_aggregator(row: pd.Series, dist_col: str = "min_agg_aug_dist", t=0.3, k=10.0):
    """
    Convert per-dialect distances dict -> corpus-level AGS using smooth gating.
    Returns {6: score_for_6_dialects, 26: score_for_26_dialects (if applicable)}.
    """
    distances = row.get(dist_col, {}) or {}
    corpus = row.get("corpus", 6)

    vals = {d: None for d in distances.keys()}
    for dialect, distance in distances.items():
        if distance is None:
            distance = 1.0
        vals[dialect] = smooth_threshold(distance, t, k)

    score6 = sum(vals.get(d, 0.0) for d in dialects_6) / len(dialects_6)

    if corpus == 6:
        return {6: score6}
    # corpus == 26
    score26 = sum(vals.values()) / max(len(vals), 1)
    return {6: score6, 26: score26}

def unpack_scores(df: pd.DataFrame, col: str):
    """Split dict column {6: v6, 26: v26} into two numeric columns, keep original."""
    df[col + "_c6"] = df[col].map(lambda v: (v or {}).get(6) if isinstance(v, dict) else None)
    df[col + "_c26"] = df[col].map(lambda v: (v or {}).get(26) if isinstance(v, dict) else None)

def compute_distances(df: pd.DataFrame,
                      distance_calculator: DistanceCalculator,
                      dialects: list,
                      aggregators=("min", "freq_softmax"),
                      preprocess: bool = True,
                      temperature: float = 5.0) -> pd.DataFrame:
    """
    For each row (source word & dialect), compute per-target-dialect aggregated augmented distances.
    Adds two dict-columns (if requested): 'min_agg_aug_dist' and 'softmax_inverse_agg_aug_dist'.
    """
    if "word" not in df.columns or "dialect" not in df.columns:
        raise ValueError("DataFrame must contain columns: 'word' and 'dialect'.")

    # init requested aggregator columns
    if "min" in aggregators:
        df["min_agg_aug_dist"] = df.apply(lambda _: {d: None for d in dialects}, axis=1)
    if "freq_softmax" in aggregators:
        df["softmax_inverse_agg_aug_dist"] = df.apply(lambda _: {d: None for d in dialects}, axis=1)

    for i, row in df.iterrows():
        src_word = row["word"]
        src_dial = row["dialect"]

        for tgt_dial in dialects:
            tgt_map_col = tgt_dial
            if src_dial == tgt_dial:
                if "min" in aggregators:
                    df.at[i, "min_agg_aug_dist"][tgt_dial] = 0.0
                if "freq_softmax" in aggregators:
                    df.at[i, "softmax_inverse_agg_aug_dist"][tgt_dial] = 0.0
                continue

            word_dict = row.get(tgt_map_col)
            if not (isinstance(word_dict, dict) and len(word_dict) > 0):
                # no aligned candidates
                if "min" in aggregators:
                    df.at[i, "min_agg_aug_dist"][tgt_dial] = None
                if "freq_softmax" in aggregators:
                    df.at[i, "softmax_inverse_agg_aug_dist"][tgt_dial] = None
                continue

            distances, freqs = [], []
            for tgt_word, freq in word_dict.items():
                if tgt_word is None:
                    continue
                res = distance_calculator.augmented_distance(
                    {
                        "source": src_word,
                        "target": tgt_word,
                        "source_dialect": src_dial,
                        "target_dialect": tgt_dial,
                    },
                    preprocess=preprocess,
                    verbose=False,
                )
                distances.append(res["normalized_distance"])
                freqs.append(freq)

            # write aggregations
            if "min" in aggregators:
                df.at[i, "min_agg_aug_dist"][tgt_dial] = min(distances) if distances else None
            if "freq_softmax" in aggregators:
                df.at[i, "softmax_inverse_agg_aug_dist"][tgt_dial] = (
                    freq_inverse_softmax_aggregate(distances, freqs, temperature=temperature)
                    if distances else None
                )
    return df

# ----------------- main -----------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) load substitution resources (from output/)
    load_distance_resources()

    # 2) load alignments
    if not os.path.exists(ALIGN_PATH):
        raise FileNotFoundError(f"Missing alignments file: {ALIGN_PATH}")
    df = pd.read_csv(ALIGN_PATH, sep="\t")

    # dialect columns are dict-like strings → real dicts
    present_dial_cols = [d for d in dialects_26 if d in df.columns]
    if present_dial_cols:
        df.loc[:, present_dial_cols] = df.loc[:, present_dial_cols].map(ast.literal_eval)

    # 3) mark corpus (6 vs 26)
    extra_dialects = set(dialects_26) - set(dialects_6)
    df["corpus"] = df.apply(lambda r: determine_corpus(r, extra_dialects), axis=1)

    # 4) build distance calculator (minimal; affixes not required for AGS)
    calc = DistanceCalculator(
        compute_substitution_cost_func=compute_substitution_cost,
        scaler="exponential",
        threshold=0,
        prefixes={},  # optional, not needed for AGS scoring
        suffixes={},
    )

    # 5) compute per-dialect aggregated distances
    df = compute_distances(
        df,
        distance_calculator=calc,
        dialects=present_dial_cols or dialects_26,
        aggregators=("min", "freq_softmax"),
        preprocess=True,
        temperature=5.0,
    )

    # 6) smooth threshold → AGS (two families + three thresholds)
    # min-agg
    df["min_t_0.3_k_20"] = df.apply(lambda x: smooth_threshold_aggregator(x, dist_col="min_agg_aug_dist", t=0.3, k=20), axis=1)
    df["min_t_0.4_k_20"] = df.apply(lambda x: smooth_threshold_aggregator(x, dist_col="min_agg_aug_dist", t=0.4, k=20), axis=1)
    df["min_t_0.5_k_20"] = df.apply(lambda x: smooth_threshold_aggregator(x, dist_col="min_agg_aug_dist", t=0.5, k=20), axis=1)
    # softmax-agg
    df["softmax_t_0.3_k_20"] = df.apply(lambda x: smooth_threshold_aggregator(x, dist_col="softmax_inverse_agg_aug_dist", t=0.3, k=20), axis=1)
    df["softmax_t_0.4_k_20"] = df.apply(lambda x: smooth_threshold_aggregator(x, dist_col="softmax_inverse_agg_aug_dist", t=0.4, k=20), axis=1)
    df["softmax_t_0.5_k_20"] = df.apply(lambda x: smooth_threshold_aggregator(x, dist_col="softmax_inverse_agg_aug_dist", t=0.5, k=20), axis=1)

    # 7) unpack 6/26 views to flat numeric columns (keep dicts too)
    agg_cols = [
        "min_t_0.3_k_20", "min_t_0.4_k_20", "min_t_0.5_k_20",
        "softmax_t_0.3_k_20", "softmax_t_0.4_k_20", "softmax_t_0.5_k_20",
    ]
    for c in agg_cols:
        unpack_scores(df, c)

    # 7b) write the full table (dict columns + unpacked c6/c26) for training-data construction
    df.to_csv(FULL_OUT, sep="\t", index=False)
    print(f"[OK] full aug/agg table written to: {FULL_OUT}")

    # 8) save compact result
    keep = ["word", "dialect", "corpus"] if all(k in df.columns for k in ["word", "dialect", "corpus"]) else ["corpus"]
    keep += [c + "_c6" for c in agg_cols] + [c + "_c26" for c in agg_cols]
    # if some keep-cols are missing (e.g., corpus==6 for all), filter gracefully
    keep = [c for c in keep if c in df.columns]

    df_to_save = df[keep].copy()
    df_to_save.to_csv(SCORES_OUT, sep="\t", index=False)
    print(f"[OK] AGS scores written to: {SCORES_OUT}")

if __name__ == "__main__":
    main()
