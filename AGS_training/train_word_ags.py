#!/usr/bin/env python3
"""
Fine-tune the word-only AGS regressor (``Exp_1.ipynb``, cells 20-25).

Input:  a ``word,generality`` CSV from
        ``AGS_training/build_training_data.py --word-only``.
Model:  ``AutoModelForSequenceClassification`` (num_labels=1) over the string
        ``"<GENERALITY> {word}"``, MSE loss, AdamW(lr=2e-5), max_length=20, 3 epochs.
Output: ``pytorch_model.bin`` via ``save_pretrained`` per epoch, plus
        ``dev_metrics_log.txt``.

wandb logging is opt-in via ``--wandb-project`` + ``WANDB_API_KEY``.

Usage (from the repo root):
  python -m AGS_training.prepare_tokenizer --mode word --out models/ags_word_tokenizer
  python -m AGS_training.build_training_data --word-only --target-col min_t_0.5_k_20
  python -m AGS_training.train_word_ags \\
    --train-csv output/ags_word_train.csv --dev-csv output/ags_word_dev.csv \\
    --tokenizer-dir models/ags_word_tokenizer --output-dir models/ags_word
"""

import argparse
import logging
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from AGS_training.prepare_tokenizer import prepare_tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("train_word_ags")

BASE_MODEL = "CAMeL-Lab/bert-base-arabic-camelbert-mix"
SPECIAL_TOKEN = "<GENERALITY>"


class WordDataset(Dataset):
    def __init__(self, words, targets, tokenizer, max_length=20):
        self.words = list(words)
        self.targets = list(targets)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        enc = self.tokenizer.encode(
            f"{SPECIAL_TOKEN} {self.words[idx]}", truncation=True,
            padding="max_length", max_length=self.max_length, return_tensors="pt",
        )
        return enc.squeeze(0), torch.tensor(self.targets[idx], dtype=torch.float32)


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _evaluate(model, loader, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            out = model(inputs).logits.squeeze(-1)
            losses.append(F.mse_loss(out, targets).item())
    model.train()
    mse = float(np.mean(losses))
    return mse, float(np.sqrt(mse))


def train(args):
    _seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    if not os.path.isdir(args.tokenizer_dir):
        logger.info("Tokenizer dir missing; building one at %s", args.tokenizer_dir)
        prepare_tokenizer("word", args.base_model, args.tokenizer_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)

    import pandas as pd
    train_df = pd.read_csv(args.train_csv)
    dev_df = pd.read_csv(args.dev_csv)
    train_loader = DataLoader(
        WordDataset(train_df["word"], train_df["generality"], tokenizer, args.max_length),
        batch_size=args.batch_size, shuffle=True,
    )
    dev_loader = DataLoader(
        WordDataset(dev_df["word"], dev_df["generality"], tokenizer, args.max_length),
        batch_size=args.batch_size,
    )

    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=1)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "dev_metrics_log.txt")

    wb = None
    if args.wandb_project:
        import wandb
        wb = wandb.init(project=args.wandb_project, name=args.wandb_run, resume="allow")

    global_step = 0
    stop = False
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for step, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            out = model(inputs).logits.squeeze(-1)
            loss = F.mse_loss(out, targets)
            loss.backward()
            optimizer.step()
            running += loss.item()
            global_step += 1
            if wb:
                wb.log({"train_loss": loss.item()}, step=global_step)

            if global_step % args.eval_steps == 0:
                dev_mse, dev_rmse = _evaluate(model, dev_loader, device)
                train_loss = running / (step + 1)
                with open(log_path, "a") as f:
                    f.write(f"Step {global_step}: Train Loss: {train_loss:.4f}, "
                            f"Dev RMSE: {dev_rmse:.4f}, Dev Loss: {dev_mse:.4f}\n")
                logger.info("step %d | train_loss %.4f | dev_rmse %.4f", global_step, train_loss, dev_rmse)
                if wb:
                    wb.log({"dev_rmse": dev_rmse, "dev_loss": dev_mse}, step=global_step)

            if args.max_steps and global_step >= args.max_steps:
                stop = True
                break
        model.save_pretrained(args.output_dir)
        logger.info("epoch %d checkpoint -> %s", epoch + 1, args.output_dir)
        if stop:
            break
    tokenizer.save_pretrained(args.output_dir)
    if wb:
        wb.finish()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-csv", required=True)
    p.add_argument("--dev-csv", required=True)
    p.add_argument("--tokenizer-dir", default="models/ags_word_tokenizer")
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument("--output-dir", default="models/ags_word")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=20)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=None, help="Stop early (smoke tests)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-run", default=None)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
