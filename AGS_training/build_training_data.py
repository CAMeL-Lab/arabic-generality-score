#!/usr/bin/env python3
"""
Turn the per-word AGS table into ``(sentence, score, dialect)`` training rows.

Reproduces the data-construction cells of ``Exp_2.ipynb`` (contextual model) and
``Exp_1.ipynb`` (word-only model):

  1. build the corpus-6 / corpus-26 views of the AGS table and flatten each score
     column to a single number (``[6]`` for the -6 view, ``[corpus]`` for the -26 view);
  2. drop multi-word entries, digits and single characters, and rows where too many
     dialect columns have no alignment (``{None: 1}``);
  3. for every surviving (word, dialect), find MADAR sentences in that dialect that
     contain the word and wrap the first occurrence with ``[TGT] ... [/TGT]``;
  4. explode 1-2 marked sentences per row and stratified-split 90/10 by dialect.

Input:  output/MADAR_26_word_alignment_aug_agg.tsv  (written by AGS_extraction/AGS_extraction.py)
        data/MADAR/MADAR.tsv                        (written by MADAR_alignment/build_madar_table.py)
Output (contextual): <out-dir>/ags_train_<corpus>.csv, ags_dev_<corpus>.csv
Output (word-only):  <out-dir>/ags_word_train.csv, ags_word_dev.csv   (with --word-only)

Usage (from the repo root):
  python -m AGS_training.build_training_data --corpus 6 --target-col min_t_0.5_k_20
  python -m AGS_training.build_training_data --corpus 26 --target-col min_t_0.5_k_20
  python -m AGS_training.build_training_data --word-only --target-col min_t_0.5_k_20

(the thesis calls the min-aggregator family ``smooth_t_*``; AGS_extraction.py writes
it as ``min_t_*`` — same quantity.)
"""

import argparse
import ast
import logging
import os
import random
import re
from collections import defaultdict

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("build_training_data")

DEFAULT_AGG = os.path.join("output", "MADAR_26_word_alignment_aug_agg.tsv")
DEFAULT_MADAR = os.path.join("data", "MADAR", "MADAR.tsv")
DEFAULT_OUT_DIR = "output"

# Corpus-6 uses unambiguous codes. Corpus-26 uses the codes that the rest of this
# repo (reformat_alignments.py / AGS_extraction.py) actually writes as columns
# (ALEX/ARI/SFX). See KNOWN_ISSUES.md about the historical ALX/RIY/SFA spelling.
DIALECTS_6 = ["MSA", "BEI", "CAI", "TUN", "DOH", "RAB"]
DIALECTS_26 = [
    "BEI", "ALEX", "AMM", "ASW", "ALE", "CAI", "DAM", "JER", "SAL", "MSA", "DOH",
    "RAB", "TUN", "ALG", "BAG", "BAS", "BEN", "FES", "JED", "KHA", "MOS", "MUS",
    "ARI", "SAN", "SFX", "TRI",
]
SCORE_COLS = [
    "min_t_0.3_k_20", "min_t_0.4_k_20", "min_t_0.5_k_20",
    "softmax_t_0.3_k_20", "softmax_t_0.4_k_20", "softmax_t_0.5_k_20",
]

_NP_FLOAT_RE = re.compile(r"np\.float64\(([\d\.eE+-]+)\)")


def _safe_eval_cell(v):
    """Parse the ``repr()``-serialised dict/list cells written by to_csv."""
    if isinstance(v, str) and v[:1] in "{[":
        return ast.literal_eval(_NP_FLOAT_RE.sub(r"\1", v))
    return v


def _has_only_none_key(d) -> bool:
    return isinstance(d, dict) and list(d.keys()) == [None]


def mark_target_word(sentence: str, target_word: str):
    """Wrap the first substring occurrence of ``target_word`` (Exp_2 cell 19)."""
    if target_word in sentence:
        return sentence.replace(target_word, f"[TGT]{target_word}[/TGT]", 1)
    return None


def load_agg_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df = df.map(_safe_eval_cell)
    return df


def _flatten_scores(df: pd.DataFrame, corpus: int) -> pd.DataFrame:
    """corpus==6 -> score[6]; corpus==26 -> score[row['corpus']] (Exp_2 cells 10-11)."""
    df = df.copy()
    if corpus == 6:
        df = df[df["dialect"].isin(DIALECTS_6)].copy()
        for c in SCORE_COLS:
            df[c] = df[c].map(lambda x: x[6] if isinstance(x, dict) else x)
        keep = ["word", "dialect"] + DIALECTS_6 + SCORE_COLS
    else:
        for c in SCORE_COLS:
            df[c] = df.apply(
                lambda r, c=c: r[c][r["corpus"]] if isinstance(r[c], dict) else r[c], axis=1
            )
        keep = ["word", "dialect", "corpus"] + DIALECTS_26 + SCORE_COLS
    keep = [k for k in keep if k in df.columns]
    return df[keep]


def _filter(df: pd.DataFrame, corpus: int, n_none_6: int, n_none_26: int) -> pd.DataFrame:
    """Exp_2 cell 13-14: drop MWEs / digits / 1-char words / too-many-empty rows."""
    dialect_cols = DIALECTS_6 if corpus == 6 else DIALECTS_26
    out = df[df["word"].apply(lambda x: isinstance(x, str) and len(x.split()) < 3)]
    if corpus == 6:
        out = out[out.apply(
            lambda r: sum(_has_only_none_key(r[c]) for c in dialect_cols) <= n_none_6, axis=1
        )]
    else:
        limit = {6: n_none_6, 26: n_none_26}
        out = out[out.apply(
            lambda r: sum(_has_only_none_key(r[c]) for c in dialect_cols) <= limit[r["corpus"]], axis=1
        )]
    out = out[out["word"].apply(lambda x: not x.isdigit() and len(x) > 1)]
    return out.copy()


def build_word_dialect_index(madar_df: pd.DataFrame, dialects) -> dict:
    """(word, dialect) -> [sentID] over the raw MADAR sentence table (Exp_2 cell 18)."""
    index = defaultdict(list)
    for sent_id, row in madar_df.iterrows():
        for dialect in dialects:
            sent = row.get(dialect)
            if pd.notna(sent):
                for word in set(str(sent).split()):
                    index[(word, dialect)].append(sent_id)
    return index


def add_marked_sentences(df: pd.DataFrame, madar_df: pd.DataFrame, word_dialect_index: dict) -> pd.DataFrame:
    """For each row, mark the first dict-word's occurrences in its dialect (Exp_2 cell 20)."""
    marked_col, sent_id_col = [], []
    for _, row in df.iterrows():
        dialect = row["dialect"]
        words = [w for w in list(row[dialect].keys()) if w is not None] if isinstance(row[dialect], dict) else []
        row_marked, row_ids = [], []
        for word in words:
            if (word, dialect) in word_dialect_index:
                sent_ids = word_dialect_index[(word, dialect)]
                sentences = madar_df.loc[sent_ids, dialect]
                row_marked.append([mark_target_word(s, word) for s in sentences])
                row_ids.append(list(sent_ids))
            elif " " in word:
                subwords = word.split()
                cand = [i for sw in subwords for i in word_dialect_index.get((sw, dialect), [])]
                cand_df = madar_df.loc[cand]
                hits = cand_df[cand_df[dialect].astype(str).str.contains(re.escape(word))]
                if not hits.empty:
                    row_marked.append([mark_target_word(s, word) for s in hits[dialect]])
                    row_ids.append(hits.index.tolist())
                else:
                    row_marked.append(None)
                    row_ids.append(None)
        marked_col.append(row_marked[0] if row_marked else None)
        sent_id_col.append(row_ids[0] if row_ids else None)
    df = df.copy()
    df["marked_sentence"] = marked_col
    df["sentID.BTEC"] = sent_id_col
    return df.dropna(subset=["marked_sentence"])


def explode_and_split(df: pd.DataFrame, target_col: str, seed: int, test_size: float = 0.1):
    """1-2 marked sentences per row, then stratified 90/10 split (Exp_2 cells 42-43)."""
    random.seed(seed)
    rows = []
    for _, row in df.iterrows():
        sents = [s for s in (row["marked_sentence"] or []) if s]
        if not sents:
            continue
        k = min(len(sents), random.choice([1, 2]))
        for sent in random.sample(sents, k):
            rows.append({"sentence": sent, "score": row[target_col], "dialect": row["dialect"]})
    expanded = pd.DataFrame(rows)
    try:
        train_df, dev_df = train_test_split(
            expanded, test_size=test_size, stratify=expanded["dialect"], random_state=seed
        )
    except ValueError:  # a dialect with <2 rows — only happens on tiny subsets
        logger.warning("stratified split not possible (too few rows per dialect); splitting unstratified")
        train_df, dev_df = train_test_split(expanded, test_size=test_size, random_state=seed)
    return train_df, dev_df


def build_word_only(df: pd.DataFrame, target_col: str, seed: int, test_size: float = 0.1):
    """Exp_1 cells 7-9 + 19: group the raw dialect word by the mean target score."""
    df = df.copy()
    df["unprocessed_word"] = df.apply(
        lambda r: max(r[r["dialect"]], key=r[r["dialect"]].get) if isinstance(r[r["dialect"]], dict) and r[r["dialect"]] else None,
        axis=1,
    )
    df = df.dropna(subset=["unprocessed_word"])
    grouped = df.groupby("unprocessed_word", as_index=False).agg({target_col: "mean"})
    grouped.columns = ["word", "generality"]
    grouped = grouped.sample(frac=1, random_state=seed).reset_index(drop=True)
    cut = int((1 - test_size) * len(grouped))
    return grouped[:cut], grouped[cut:]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agg-table", default=DEFAULT_AGG)
    p.add_argument("--madar-tsv", default=DEFAULT_MADAR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--corpus", type=int, choices=[6, 26], default=6)
    p.add_argument("--target-col", default="min_t_0.5_k_20",
                   help="Score column used as the regression label (one of min_t_* / softmax_t_*)")
    p.add_argument("--n-none-6", type=int, default=1)
    p.add_argument("--n-none-26", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--word-only", action="store_true", help="Build the Exp_1 word-only dataset instead")
    return p.parse_args()


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    logger.info("Loading AGS table: %s", args.agg_table)
    agg = load_agg_table(args.agg_table)

    if args.word_only:
        # Exp_1 works off the corpus-6 view.
        view = _flatten_scores(agg, 6)
        view = _filter(view, 6, args.n_none_6, args.n_none_26)
        train_df, dev_df = build_word_only(view, args.target_col, args.seed)
        train_path = os.path.join(args.out_dir, "ags_word_train.csv")
        dev_path = os.path.join(args.out_dir, "ags_word_dev.csv")
    else:
        view = _flatten_scores(agg, args.corpus)
        view = _filter(view, args.corpus, args.n_none_6, args.n_none_26)
        logger.info("Rows after filtering: %d", len(view))

        madar = pd.read_csv(args.madar_tsv, sep="\t").set_index("sentID.BTEC")
        dialects = DIALECTS_6 if args.corpus == 6 else DIALECTS_26
        logger.info("Indexing MADAR sentences by (word, dialect) ...")
        idx = build_word_dialect_index(madar, dialects)
        view = add_marked_sentences(view, madar, idx)
        logger.info("Rows with a usable marked sentence: %d", len(view))

        train_df, dev_df = explode_and_split(view, args.target_col, args.seed)
        train_path = os.path.join(args.out_dir, f"ags_train_{args.corpus}.csv")
        dev_path = os.path.join(args.out_dir, f"ags_dev_{args.corpus}.csv")

    train_df.to_csv(train_path, index=False)
    dev_df.to_csv(dev_path, index=False)
    logger.info("Wrote %d train / %d dev rows -> %s , %s", len(train_df), len(dev_df), train_path, dev_path)


if __name__ == "__main__":
    main(parse_args())
