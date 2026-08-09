#!/usr/bin/env python3
"""
Save a copy of the base CAMeLBERT tokenizer with the AGS special tokens added.

- The **sentence** model (train_sentence_ags.py) marks the target span with
  ``[TGT] ... [/TGT]``.
- The **word** model (train_word_ags.py) prefixes the bare word with ``<GENERALITY>``.

Usage (from the repo root):
  python -m AGS_training.prepare_tokenizer --mode sentence --out models/ags_tokenizer
  python -m AGS_training.prepare_tokenizer --mode word --out models/ags_word_tokenizer
"""

import argparse
import logging

from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("prepare_tokenizer")

BASE_MODEL = "CAMeL-Lab/bert-base-arabic-camelbert-mix"
SENTENCE_TOKENS = ["[TGT]", "[/TGT]"]
WORD_TOKENS = ["<GENERALITY>"]


def prepare_tokenizer(mode: str = "sentence", base_model: str = BASE_MODEL, out_dir: str = "models/ags_tokenizer"):
    special = SENTENCE_TOKENS if mode == "sentence" else WORD_TOKENS
    logger.info("Loading base tokenizer: %s", base_model)
    tok = AutoTokenizer.from_pretrained(base_model)
    added = tok.add_special_tokens({"additional_special_tokens": special})
    logger.info("Added %d special token(s): %s", added, special)
    tok.save_pretrained(out_dir)
    logger.info("Saved tokenizer to: %s (vocab size %d)", out_dir, len(tok))
    return tok


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["sentence", "word"], default="sentence")
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument("--out", default="models/ags_tokenizer", help="Directory to save the tokenizer into")
    return p.parse_args()


def main(args):
    prepare_tokenizer(args.mode, args.base_model, args.out)


if __name__ == "__main__":
    main(parse_args())
