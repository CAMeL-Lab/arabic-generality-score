# arabic-generality-score
Arabic Generality Score


## Install dependencies:

pip install -r requirements.txt

## Finetune Awesome Align on MADAR:

python MADAR_alignment/finetune_awesome_align.py

## Use finetuned awesome align to run word-alignment on MADAR:

python MADAR_alignment/run_alignment_madar.py \
  --data_file MADAR_alignment/data/AWESOME_finetuning_data.txt \
  --output_file output/finetuned_awesome_align_output_MADAR_26_idx.txt \
  --model_name_or_path models/awesome_align_finetuned_camelbert_mix \
  --batch_size 32





##### TBC

