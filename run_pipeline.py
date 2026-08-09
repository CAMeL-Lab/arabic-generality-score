#!/usr/bin/env python3
"""
Run the AGS pipeline end to end (or a slice of it).

Stages, in order:
  build-table       MADAR_alignment.build_madar_table      -> data/MADAR/MADAR.tsv
  finetune-align    MADAR_alignment.finetune_awesome_align -> models/awesome_align_finetuned_camelbert_mix   [GPU]
  run-align         MADAR_alignment.run_alignment_madar    -> output/finetuned_awesome_align_output_MADAR_26_idx.txt   [GPU]
  reformat          MADAR_alignment.reformat_alignments    -> output/MADAR_reformatted_word_alignments.tsv
  precompute-prob   distance_function.precompute_probabilities -> output/*_probabilities*.tsv, *.json
  ags-extract       AGS_extraction.AGS_extraction          -> output/AGS_scores.tsv, MADAR_26_word_alignment_aug_agg.tsv   [slow: hours on the full corpus]
  build-train-data  AGS_training.build_training_data       -> output/ags_train_6.csv, ags_dev_6.csv
  train-sentence    AGS_training.train_sentence_ags        -> models/ags_sentence/checkpoint_step*.pt   [GPU]
  evaluate          AGS_evaluation.evaluate_mdid           (needs --mdid-csv; NADI data not bundled)
  export            AGS_evaluation.export_to_hf            -> models/ags_hf_export

Examples:
  python run_pipeline.py --from reformat --to ags-extract
  python run_pipeline.py --only build-table
  python run_pipeline.py --list
"""

import argparse
import subprocess
import sys

STAGES = [
    ("build-table",      [sys.executable, "-m", "MADAR_alignment.build_madar_table"]),
    ("finetune-align",   [sys.executable, "-m", "MADAR_alignment.finetune_awesome_align"]),
    ("run-align",        [sys.executable, "-m", "MADAR_alignment.run_alignment_madar",
                          "--data_file", "MADAR_alignment/data/AWESOME_finetuning_data.txt",
                          "--output_file", "output/finetuned_awesome_align_output_MADAR_26_idx.txt",
                          "--model_name_or_path", "models/awesome_align_finetuned_camelbert_mix",
                          "--batch_size", "32"]),
    ("reformat",         [sys.executable, "-m", "MADAR_alignment.reformat_alignments"]),
    ("precompute-prob",  [sys.executable, "-m", "distance_function.precompute_probabilities"]),
    ("ags-extract",      [sys.executable, "-m", "AGS_extraction.AGS_extraction"]),
    ("build-train-data", [sys.executable, "-m", "AGS_training.build_training_data",
                          "--corpus", "6", "--target-col", "min_t_0.5_k_20"]),
    ("train-sentence",   [sys.executable, "-m", "AGS_training.train_sentence_ags",
                          "--train-csv", "output/ags_train_6.csv",
                          "--dev-csv", "output/ags_dev_6.csv",
                          "--tokenizer-dir", "models/ags_tokenizer",
                          "--output-dir", "models/ags_sentence"]),
    ("evaluate",         [sys.executable, "-m", "AGS_evaluation.evaluate_mdid"]),
    ("export",           [sys.executable, "-m", "AGS_evaluation.export_to_hf"]),
]
NAMES = [n for n, _ in STAGES]
GPU_STAGES = {"finetune-align", "run-align", "train-sentence"}
EXTERNAL_DATA_STAGES = {"evaluate"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from", dest="start", choices=NAMES, default=NAMES[0])
    p.add_argument("--to", dest="end", choices=NAMES, default="ags-extract")
    p.add_argument("--only", choices=NAMES, help="Run just this stage")
    p.add_argument("--list", action="store_true", help="Print the stage list and exit")
    p.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="Everything after --extra is appended to every stage command")
    return p.parse_args()


def main():
    args = parse_args()
    if args.list:
        for n in NAMES:
            tags = []
            if n in GPU_STAGES:
                tags.append("GPU")
            if n in EXTERNAL_DATA_STAGES:
                tags.append("needs external data")
            print(f"  {n:17s} {'(' + ', '.join(tags) + ')' if tags else ''}")
        return

    if args.only:
        selected = [args.only]
    else:
        i, j = NAMES.index(args.start), NAMES.index(args.end)
        if i > j:
            raise SystemExit(f"--from {args.start} is after --to {args.end}")
        selected = NAMES[i:j + 1]

    for name in selected:
        cmd = dict(STAGES)[name] + list(args.extra)
        marker = "  [GPU]" if name in GPU_STAGES else ("  [needs --mdid-csv]" if name in EXTERNAL_DATA_STAGES else "")
        print(f"\n=== stage: {name}{marker} ===\n$ {' '.join(cmd)}")
        if args.dry_run:
            continue
        r = subprocess.run(cmd)
        if r.returncode != 0:
            raise SystemExit(f"stage '{name}' failed (exit {r.returncode})")


if __name__ == "__main__":
    main()
