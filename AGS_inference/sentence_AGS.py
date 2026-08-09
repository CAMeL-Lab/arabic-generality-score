#!/usr/bin/env python3
"""
Sentence-level AGS inference.

Scores each token by wrapping one occurrence at a time with [TGT]...[/TGT],
then aggregates with the Generalized Harmonic Mean (GHM).

Example:
  python -m AGS_inference.sentence_AGS \
    --text "هذا مثال بسيط لقياس العمومية على مستوى الجملة." \
    --p 2.0
"""

from __future__ import annotations  # `str | None` / `tuple[...]` hints on Python 3.9

import argparse
import re
from typing import List, Tuple
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

HF_REPO = "Sanadshabann/AGS"

def simple_ar_word_split(text: str) -> List[str]:
    # super minimal: split by whitespace
    return [t for t in text.split() if t.strip()]

@torch.inference_mode()
def predict_word_scores(sentence: str, tok, model, device: str | None = None, max_words: int = 64) -> Tuple[List[str], List[float]]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    words = simple_ar_word_split(sentence)[:max_words]
    if not words:
        return [], []
    # mark first occurrence of each word in order
    marked_variants = [
        re.sub(rf"\b{re.escape(w)}\b", f"[TGT]{w}[/TGT]", sentence, count=1)
        for w in words
    ]
    enc = tok(marked_variants, return_tensors="pt", truncation=True, padding=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    logits = model.to(device)(**enc).logits.squeeze(-1)  # [batch]
    scores = [float(x) for x in logits.detach().cpu()]
    return words, scores

def aggregate_sentence_score(word_scores, p: float = 2.0, eps: float = 1e-8) -> float:
    # GHM_p = ( mean( (x_i + eps)^(-p) ) )^(-1/p)
    arr = np.array(word_scores, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    ghm = (np.mean((arr + eps) ** (-p))) ** (-1.0 / p)
    return float(ghm)

def predict_sentence(text: str, repo: str = HF_REPO, p: float = 2.0) -> tuple[list[str], list[float], float]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(repo, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(repo).to(device).eval()
    words, scores = predict_word_scores(text, tok, model, device=device)
    sent_score = aggregate_sentence_score(scores, p=p)
    return words, scores, sent_score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True, help="Raw sentence (no [TGT] tags)")
    ap.add_argument("--repo", default=HF_REPO, help="HF repo id (default: Sanadshabann/AGS)")
    ap.add_argument("--p", type=float, default=2.0, help="GHM exponent (p>1 emphasizes low scores)")
    args = ap.parse_args()

    words, scores, sent = predict_sentence(args.text, repo=args.repo, p=args.p)
    # pretty print
    preview_words = " ".join(words[:10]) + (" ..." if len(words) > 10 else "")
    preview_scores = ", ".join(f"{s:.4f}" for s in scores[:10]) + (" ..." if len(scores) > 10 else "")
    print("words:", preview_words)
    print("word_scores:", preview_scores)
    print(f"sentence_score (GHM, p={args.p:g}): {sent:.6f}")

if __name__ == "__main__":
    main()
