# Arabic Generality Score (AGS)

Reference implementation of **"The Arabic Generality Score: Another Dimension of
Modeling Arabic Dialectness"** (Sha'ban & Habash, EMNLP 2025).

- Paper: https://aclanthology.org/2025.emnlp-main.1524/
- Trained model: https://huggingface.co/Sanadshabann/AGS

AGS measures how *general* (understood across dialects) vs. *dialect-specific* an
Arabic word is. It is derived by aligning MADAR across 26 dialects, measuring a
phonology-aware "augmented" edit distance between each word and its cross-dialect
equivalents, gating those distances with a smooth threshold, and finally distilling
the per-word scores into a CAMeLBERT regressor that predicts AGS for a word in
context.

## Layout

| Stage | Module | Produces |
|---|---|---|
| 0. Build the MADAR-26 table | `MADAR_alignment/build_madar_table.py` | `data/MADAR/MADAR.tsv` |
| 1. Fine-tune the aligner | `MADAR_alignment/finetune_awesome_align.py` | `models/awesome_align_finetuned_camelbert_mix/` |
| 2. Word-align MADAR-26 | `MADAR_alignment/run_alignment_madar.py` | `output/finetuned_awesome_align_output_MADAR_26_idx.txt` |
| 3. Reformat alignments | `MADAR_alignment/reformat_alignments.py` | `output/MADAR_reformatted_word_alignments.tsv` |
| 4. Probability tables | `distance_function/precompute_probabilities.py` | `output/*_probabilities*.tsv`, `P_CAPHI_given_ORTHO_by_dialect.json` |
| — the distance itself | `distance_function/substitution_weight.py`, `augmented_edit_distance.py` | (library) |
| 5. Extract per-word AGS | `AGS_extraction/AGS_extraction.py` | `output/AGS_scores.tsv`, `MADAR_26_word_alignment_aug_agg.tsv` |
| 6. Build training data | `AGS_training/build_training_data.py` | `output/ags_{train,dev}_{6,26}.csv` |
| 7. Train the regressor | `AGS_training/train_sentence_ags.py` (contextual), `train_word_ags.py` (word-only) | `models/ags_*/checkpoint_step*.pt` |
| 8. Inference | `AGS_inference/word_AGS.py`, `sentence_AGS.py` | scores |
| 9. Evaluate / export | `AGS_evaluation/evaluate_mdid.py`, `baselines.py`, `export_to_hf.py` | RMSE tables, HF model dir |

`utilities/preprocess_text.py` is the shared Arabic normalizer.
`run_pipeline.py` chains stages 0–9.

## Install

```bash
python -m pip install -e .          # or: pip install -r requirements.txt
camel_data -i light                 # CAMeL Tools morphology/normalization data
```

`awesome-align` (stages 1–2) is a separate package with its own heavy deps and needs
a GPU; it is listed in `requirements.txt` but you may prefer to install it in its own
environment. Stages 1, 2 and 7 need a GPU to be practical.

## Run the pipeline

```bash
python run_pipeline.py --list                     # show stages
python run_pipeline.py --from reformat --to ags-extract
python run_pipeline.py --only build-table
```

or invoke the stages directly:

```bash
# 0. merge the per-city MADAR files (data/MADAR/MADAR.tsv is already committed)
python -m MADAR_alignment.build_madar_table

# 1. fine-tune AWESOME-align on MSA|||DIALECT pairs                        [GPU]
python -m MADAR_alignment.finetune_awesome_align

# 2. word-align MADAR-26 with the fine-tuned checkpoint                    [GPU]
python -m MADAR_alignment.run_alignment_madar \
  --data_file MADAR_alignment/data/AWESOME_finetuning_data.txt \
  --output_file output/finetuned_awesome_align_output_MADAR_26_idx.txt \
  --model_name_or_path models/awesome_align_finetuned_camelbert_mix \
  --batch_size 32

# 3. pivot around MSA, resolve many-to-one, aggregate to a per-concept table
python -m MADAR_alignment.reformat_alignments \
  --alignment_idx_path output/finetuned_awesome_align_output_MADAR_26_idx.txt \
  --id_dialect_path output/id_dialect.txt \
  --out_tsv output/MADAR_reformatted_word_alignments.tsv

# 4. CODA/CAPHI/ORTHO probability tables (reads data/, writes output/)
python -m distance_function.precompute_probabilities

# 5. per-word AGS at t in {0.3,0.4,0.5}, k=20, for corpus-6 and corpus-26  [slow: hours]
python -m AGS_extraction.AGS_extraction
```

`AGS_OUTPUT_DIR` (default `./output`) redirects where stages 3–6 read and write.
See `.env.example`.

## Train

```bash
python -m AGS_training.prepare_tokenizer --mode sentence --out models/ags_tokenizer
python -m AGS_training.build_training_data --corpus 6 --target-col min_t_0.5_k_20
python -m AGS_training.train_sentence_ags \
  --train-csv output/ags_train_6.csv --dev-csv output/ags_dev_6.csv \
  --tokenizer-dir models/ags_tokenizer --output-dir models/ags_sentence_madar6      # [GPU]
```

The word-only variant (`Exp_1`): `build_training_data --word-only` then
`train_word_ags`. wandb logging is off unless you pass `--wandb-project` and set
`WANDB_API_KEY`.

## Inference

```bash
python -m AGS_inference.word_AGS --text "هذا مثال مع [TGT]الكلمة[/TGT] الهدف داخل الجملة."
python -m AGS_inference.sentence_AGS --text "قلبي تعب منهم" --p 2.0
```

By default these pull `Sanadshabann/AGS` from the Hub.

## Evaluate

The MDID CSV (`NADI2024_full_extended.csv`) is **not** bundled — see `data/README.md`.

```bash
python -m AGS_evaluation.evaluate_mdid --mdid-csv NADI2024_full_extended.csv \
  --tokenizer-dir models/ags_tokenizer \
  --checkpoint models/ags_sentence_madar6/checkpoint_step5500.pt --split both --k 2
python -m AGS_evaluation.baselines --mdid-csv NADI2024_full_extended.csv --baseline all
python -m AGS_evaluation.export_to_hf \
  --checkpoint models/ags_sentence_madar6/checkpoint_step5000.pt \
  --tokenizer-dir models/ags_tokenizer --out models/ags_hf_export
```


## Citation

```bibtex
@inproceedings{shaban-habash-2025-arabic,
    title = "The {A}rabic Generality Score: Another Dimension of Modeling {A}rabic Dialectness",
    author = "Sha{'}ban, Sanad  and Habash, Nizar",
    editor = "Christodoulopoulos, Christos  and Chakraborty, Tanmoy  and Rose, Carolyn  and Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov, year = "2025", address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.1524/",
    pages = "29990--30001", ISBN = "979-8-89176-332-6"
}
```
