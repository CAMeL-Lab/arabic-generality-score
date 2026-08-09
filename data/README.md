# Data

## Bundled in this repo

| Path | What it is | Upstream / licence |
|---|---|---|
| `data/MADAR/MADAR.Parallel-Corpora-Public-Version1.1-25MAR2021/` | MADAR parallel corpus, 25 city dialects + MSA (public split). | `LICENSE.txt` / `README.txt` in that folder. |
| `data/MADAR/MADAR.tsv` | The 26 city files merged on `sentID.BTEC` (one column per dialect code). Regenerate with `python -m MADAR_alignment.build_madar_table`. | derived from the above |
| `distance_function/data/MADAR-CODA/` | MADAR-CODA raw↔CODA sentence pairs (train/test) for BEI, CAI, DOH, RAB, TUN. | `LICENSE.txt` / `README.txt` in that folder |
| `distance_function/data/MADAR-CODA/word_alignment/` | Character/word alignments of the raw↔CODA pairs, produced with CAMeL-Lab's `ced_word_alignment` (`align_text.py`). Consumed directly by `precompute_probabilities.py`. | derived |
| `distance_function/data/MADAR_Lexicon_v1.0/` | MADAR Lexicon (CODA + CAPHI pronunciations per dialect). `*_aligned.tsv` adds a CODA↔CAPHI character alignment column. | `LICENSE.txt` / `README.txt` in that folder |
| `distance_function/data/CAPHI/CAPHI-table.tsv` | Letter → CAPHI phoneme table (default vs. etymological mappings). | CAPHI project |
| `MADAR_alignment/data/AWESOME_finetuning_data.txt` | `MSA ||| DIALECT` pairs for fine-tuning AWESOME-align (built by `finetune_awesome_align.py`). | derived from MADAR |
| `MADAR_alignment/data/id_dialect.txt` | `row_id<TAB>DIALECT` parallel to the file above. | derived |

## NOT bundled — needed only for evaluation

`AGS_evaluation/evaluate_mdid.py` and `AGS_evaluation/baselines.py` expect a CSV
passed via `--mdid-csv`:

- **`NADI2024_full_extended.csv`** ("MDID"): NADI-2024 subtask-1 sentences with
  per-country validity annotations. Columns used: `sentence`, `n_valid` (gold
  generality = `n_valid / 11`), `split` (`dev` / `test`). Build it from the
  NADI-2024 shared-task release (licence-restricted — request access from the NADI
  organisers) plus the MDID annotations.
