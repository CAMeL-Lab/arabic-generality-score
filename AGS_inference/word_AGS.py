#!/usr/bin/env python3
"""
Word-level AGS inference.

Given a sentence with a single target span wrapped by [TGT] ... [/TGT],
returns a single float (generality score).

Example:
  python -m AGS_inference.word_AGS \
    --text "هذا مثال مع [TGT]الكلمة[/TGT] الهدف داخل الجملة."
"""

from __future__ import annotations  # `str | None` hints on Python 3.9

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

HF_REPO = "Sanadshabann/AGS"

def predict_word(text: str, repo: str = HF_REPO, device: str | None = None) -> float:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(repo, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(repo).to(device).eval()
    with torch.inference_mode():
        enc = tok(text, return_tensors="pt", truncation=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        score = model(**enc).logits.squeeze().item()
    return float(score)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True, help="Sentence with [TGT]...[/TGT] around the target word/span")
    p.add_argument("--repo", default=HF_REPO, help="HF repo id (default: Sanadshabann/AGS)")
    args = p.parse_args()

    score = predict_word(args.text, repo=args.repo)
    print(f"{score:.6f}")

if __name__ == "__main__":
    main()
