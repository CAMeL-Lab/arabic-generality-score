#!/usr/bin/env python3
"""
Convert a ``.pt`` BertRegressor checkpoint into a standard Hugging Face
``AutoModelForSequenceClassification`` regression model
(``Evaluation-Inference.ipynb`` cell 64), optionally pushing it to the Hub.

The custom head (``regressor.*`` on the pooled [CLS]) is remapped onto HF's
``classifier.*`` so the exported model runs with plain
``AutoModelForSequenceClassification.from_pretrained(...)`` — exactly how
``AGS_inference/word_AGS.py`` and ``sentence_AGS.py`` load ``Sanadshabann/AGS``.

Usage (from the repo root):
  python -m AGS_evaluation.export_to_hf \\
    --checkpoint models/ags_sentence_madar6/checkpoint_step5000.pt \\
    --tokenizer-dir models/ags_tokenizer \\
    --out models/ags_hf_export
  # add --push --repo <user>/<name>  (needs HF_TOKEN in the environment)
"""

import argparse
import json
import logging
import os
from datetime import datetime, timezone

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("export_to_hf")

BASE_MODEL = "CAMeL-Lab/bert-base-arabic-camelbert-mix"


def _strip_prefix(k: str) -> str:
    for p in ("module.", "model.", "net."):
        if k.startswith(p):
            return k[len(p):]
    return k


def convert(checkpoint: str, tokenizer_dir: str, out_dir: str, base_model: str = BASE_MODEL):
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)
    for t in ("[TGT]", "[/TGT]"):
        if t not in tokenizer.get_vocab():
            tokenizer.add_tokens([t])

    config = AutoConfig.from_pretrained(base_model)
    config.num_labels = 1
    config.problem_type = "regression"
    config.id2label = {0: "generality"}
    config.label2id = {"generality": 0}
    hf_model = AutoModelForSequenceClassification.from_pretrained(base_model, config=config)
    hf_model.resize_token_embeddings(len(tokenizer))

    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    remapped = {}
    for k, v in state.items():
        nk = _strip_prefix(k).replace("regressor.", "classifier.")
        if isinstance(v, torch.Tensor) and not v.is_contiguous():
            v = v.contiguous()
        remapped[nk] = v
    missing, unexpected = hf_model.load_state_dict(remapped, strict=False)
    logger.info("loaded state_dict: missing=%d unexpected=%d", len(missing), len(unexpected))
    if missing:
        logger.info("  missing (first 10): %s", missing[:10])
    if unexpected:
        logger.info("  unexpected (first 10): %s", unexpected[:10])

    # smoke test
    hf_model.eval()
    with torch.inference_mode():
        _ = hf_model(**tokenizer("اختبار [TGT]كلمة[/TGT] في جملة.", return_tensors="pt", truncation=True))

    cleaned = {k: (v.contiguous() if isinstance(v, torch.Tensor) and not v.is_contiguous() else v)
               for k, v in hf_model.state_dict().items()}
    hf_model.to("cpu").save_pretrained(out_dir, state_dict=cleaned)
    tokenizer.save_pretrained(out_dir)

    with open(os.path.join(out_dir, "training_args.json"), "w", encoding="utf-8") as f:
        json.dump({
            "exported_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "task": "regression (generality score)",
            "language": "ar",
            "base_model": base_model,
            "source_checkpoint": os.path.basename(checkpoint),
            "uses_special_tokens": ["[TGT]", "[/TGT]"],
        }, f, ensure_ascii=False, indent=2)

    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(_MODEL_CARD.format(base_model=base_model))
    logger.info("exported HF model -> %s", out_dir)
    return out_dir


def push(out_dir: str, repo_id: str, private: bool = False):
    from huggingface_hub import HfApi, create_repo, upload_folder

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Set HF_TOKEN in the environment to push.")
    create_repo(repo_id, token=token, private=private, exist_ok=True)
    upload_folder(repo_id=repo_id, folder_path=out_dir, token=token,
                  commit_message="Export AGS regression checkpoint")
    logger.info("pushed -> https://huggingface.co/%s", repo_id)
    _ = HfApi  # keep import meaningful for older hub versions


_MODEL_CARD = """---
language:
- ar
library_name: transformers
pipeline_tag: text-classification
tags:
- regression
- arabic
- dialectness
- AGS
base_model: {base_model}
---

# Arabic Generality Score (AGS)

Predicts a continuous **generality** score for a target word in context. Wrap the
target span with `[TGT] ... [/TGT]`.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
tok = AutoTokenizer.from_pretrained("<repo>", use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained("<repo>").eval()
text = "هذا مثال مع [TGT]الكلمة[/TGT] الهدف داخل الجملة."
with torch.inference_mode():
    score = model(**tok(text, return_tensors="pt", truncation=True)).logits.squeeze().item()
```

`problem_type=regression`, `num_labels=1`. Evaluated with RMSE. See
https://aclanthology.org/2025.emnlp-main.1524/
"""


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer-dir", default="models/ags_tokenizer")
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument("--out", default="models/ags_hf_export")
    p.add_argument("--push", action="store_true")
    p.add_argument("--repo", default=None, help="Target Hub repo id (with --push)")
    p.add_argument("--private", action="store_true")
    return p.parse_args()


def main(args):
    out_dir = convert(args.checkpoint, args.tokenizer_dir, args.out, args.base_model)
    if args.push:
        if not args.repo:
            raise SystemExit("--push requires --repo <user>/<name>")
        push(out_dir, args.repo, args.private)


if __name__ == "__main__":
    main(parse_args())
