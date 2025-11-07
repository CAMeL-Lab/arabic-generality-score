# Arabic Generality Score (AGS) 

paper: https://aclanthology.org/2025.emnlp-main.1524/

## Install dependencies:

pip install -r requirements.txt


## Full Pipeline to Annotate MADAR-26 with AGS:


### Finetune Awesome Align on MADAR:

python MADAR_alignment/finetune_awesome_align.py

### Use finetuned awesome align to run word-alignment on MADAR:

python MADAR_alignment/run_alignment_madar.py \
  --data_file MADAR_alignment/data/AWESOME_finetuning_data.txt \
  --output_file output/finetuned_awesome_align_output_MADAR_26_idx.txt \
  --model_name_or_path models/awesome_align_finetuned_camelbert_mix \
  --batch_size 32

### Reformat alignments to a MADAR-26 word⇄word table
Creates a tidy, per-concept alignment table used by downstream steps:

python MADAR_alignment/reformat_alignments.py \
  --alignment_idx_path output/finetuned_awesome_align_output_MADAR_26_idx.txt \
  --id_dialect_path MADAR_alignment/data/id_dialect.txt \
  --out_tsv output/MADAR_reformatted_word_alignments.tsv

### Compute probability tables (RAW↔CODA, CAPHI↔CODA, CAPHI↔ORTHO)

Writes all probability artifacts to ./output:

python distance_function/compute_probabilities.py


### AGS computation:
python AGS_extraction.py \
    --output_file_path 




## For Inference using the best performing model (CAMeLBERT trained on AGS-annotated MADAR-26), refer to:

https://huggingface.co/Sanadshabann/AGS



## Citation

If you use our work please cite:

 @inproceedings{shaban-habash-2025-arabic,
    title = "The {A}rabic Generality Score: Another Dimension of Modeling {A}rabic Dialectness",
    author = "Sha{'}ban, Sanad  and
      Habash, Nizar",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.1524/",
    pages = "29990--30001",
    ISBN = "979-8-89176-332-6"
}