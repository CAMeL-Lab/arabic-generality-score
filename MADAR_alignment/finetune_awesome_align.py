import os
import sys
import logging
import subprocess
import pandas as pd
from transformers import AutoModel, AutoTokenizer

# Make project root importable so `utilities` resolves when running this file directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utilities.preprocess_text import preprocess_text  # noqa: E402

# -------------------------
# Logging setup (simple + helpful)
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# MADAR dialect columns (26)
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
    MADAR_df[dialects_26] = MADAR_df[dialects_26].applymap(
        lambda x: preprocess_text(x) if pd.notnull(x) else x
    )

    logger.info(f"MADAR loaded. Rows: {len(MADAR_df)}")
    return MADAR_df


def save_camelbert_to_local_for_finetuning(path: str = "./models/camelbert_mix"):
    """
    Download CAMeL-Lab/bert-base-arabic-camelbert-mix and save locally.
    Ensures parameters are contiguous to avoid rare serialization issues.
    """
    model_name = "CAMeL-Lab/bert-base-arabic-camelbert-mix"
    camelbert_path = path

    logger.info(f"Downloading base model: {model_name}")
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Ensure all tensors are contiguous (defensive)
    for _, param in model.named_parameters():
        if not param.is_contiguous():
            param.data = param.data.contiguous()

    os.makedirs(camelbert_path, exist_ok=True)
    logger.info(f"Saving model + tokenizer to: {camelbert_path}")
    # Use .bin format (awesome-align is okay with either, but you forced bin)
    model.save_pretrained(camelbert_path, safe_serialization=False)
    tokenizer.save_pretrained(camelbert_path)

    logger.info(f"✅ Model saved to {camelbert_path} (bin format)")


def prepare_awesome_align_finetuning_file():
    """
    Create a parallel corpus for awesome-align training:
      - Writes: ./MADAR_alignment/data/AWESOME_finetuning_data.txt  (MSA ||| DIALECT)
      - Writes: ./MADAR_alignment/data/id_dialect.txt              (row_id \t DIALECT)
    Skips rows where MSA is null/empty. Includes all dialects != 'MSA' when not null.
    """
    MADAR_df = import_MADAR()

    out_dir = "./MADAR_alignment/data"
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, "AWESOME_finetuning_data.txt")
    sent_id_dialect_path = os.path.join(out_dir, "id_dialect.txt")

    logger.info(f"Writing awesome-align training data to: {output_path}")
    total_pairs = 0
    skipped_rows_msa_empty = 0

    with open(output_path, "w", encoding="utf-8") as file, open(sent_id_dialect_path, "w", encoding="utf-8") as sent_id_file:
        for idx, row in MADAR_df.iterrows():
            msa = row["MSA"] if pd.notnull(row["MSA"]) else None
            if not msa or not str(msa).strip():
                skipped_rows_msa_empty += 1
                continue

            for dialect in dialects_26:
                if dialect == "MSA":
                    continue
                val = row[dialect]
                if pd.notnull(val) and str(val).strip():
                    file.write(f"{msa} ||| {val}\n")
                    sent_id_file.write(f"{idx}\t{dialect}\n")
                    total_pairs += 1

    logger.info(f"✅ Finished writing parallel data. Pairs: {total_pairs:,}")
    if skipped_rows_msa_empty:
        logger.warning(f"Skipped rows due to empty/null MSA: {skipped_rows_msa_empty:,}")

    return output_path, sent_id_dialect_path, total_pairs


def finetune_awesome_align():
    """
    Launch awesome-align fine-tuning via its CLI module. Assumes:
      - Training / eval files exist (created by prepare_awesome_align_finetuning_file)
      - Local CAMeL BERT exists at ./models/camelbert_mix
      - awesome-align is installed in the current venv
    """
    TRAIN_FILE = "MADAR_alignment/data/AWESOME_finetuning_data.txt"
    EVAL_FILE = "MADAR_alignment/data/AWESOME_finetuning_data.txt"
    OUTPUT_DIR = "models/awesome_align_finetuned_camelbert_mix"
    MODEL_NAME_OR_PATH = "models/camelbert_mix"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cmd = [
        sys.executable, "-m", "awesome_align.run_train",
        "--output_dir", OUTPUT_DIR,
        "--model_name_or_path", MODEL_NAME_OR_PATH,
        "--extraction", "softmax",
        "--do_train",
        "--train_tlm",
        "--train_tlm_full",
        "--train_so",
        "--train_psi",
        "--train_data_file", TRAIN_FILE,
        "--per_gpu_train_batch_size", "16",
        "--gradient_accumulation_steps", "2",
        "--num_train_epochs", "2",
        "--learning_rate", "2e-5",
        "--save_steps", "1000",
        "--max_steps", "40000",
        "--do_eval",
        "--eval_data_file", EVAL_FILE,
        "--cache_dir", OUTPUT_DIR,
    ]

    # Helpful log of the exact command being run
    logger.info("Starting awesome-align fine-tuning with command:\n%s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        logger.info("✅ awesome-align training completed successfully.")
    except FileNotFoundError:
        logger.error(
            "awesome-align is not installed or not visible in this environment.\n"
            "Try: pip install awesome-align\n"
            "Or run manually:\n  %s",
            " ".join(cmd),
        )
        raise
    except subprocess.CalledProcessError as e:
        logger.exception("awesome-align exited with non-zero code (%s).", e.returncode)
        raise


if __name__ == "__main__":
    # 1) Save the base CAMeL-BERT locally (for awesome-align finetuning)
    save_camelbert_to_local_for_finetuning()

    # 2) Build the parallel file(s) from MADAR
    prepare_awesome_align_finetuning_file()

    # 3) Launch awesome-align training
    finetune_awesome_align()
