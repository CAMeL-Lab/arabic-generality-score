#!/usr/bin/env python3
"""
Fine-tune the contextual AGS regressor (``Exp_2.ipynb``, cells 44-47).

Input:  a marked-sentence CSV (``sentence,score,dialect``) from
        ``AGS_training/build_training_data.py`` (no ``--word-only``).
Model:  CAMeLBERT-mix encoder + dropout(0.3) + Linear(hidden, 1) on the pooled
        [CLS], MSE loss, AdamW(lr=4e-5), linear schedule, batch 32, 4 epochs.
Output: ``checkpoint_step{N}.pt`` (raw ``state_dict``) every --checkpoint-steps steps,
        plus ``dev_logs.csv`` (step, dev_loss, dev_rmse). This is the checkpoint format
        that ``AGS_evaluation/export_to_hf.py`` converts for the Hub.

wandb logging is opt-in: pass ``--wandb-project`` and set ``WANDB_API_KEY`` in the
environment. Nothing is logged otherwise.

Usage (from the repo root):
  python -m AGS_training.prepare_tokenizer --mode sentence --out models/ags_tokenizer
  python -m AGS_training.build_training_data --corpus 6 --target-col min_t_0.5_k_20
  python -m AGS_training.train_sentence_ags \\
    --train-csv output/ags_train_6.csv --dev-csv output/ags_dev_6.csv \\
    --tokenizer-dir models/ags_tokenizer --output-dir models/ags_sentence_madar6
"""

import argparse
import logging
import os
import random
import shutil

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertModel, BertTokenizer, get_scheduler

from AGS_training.prepare_tokenizer import prepare_tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("train_sentence_ags")

BASE_MODEL = "CAMeL-Lab/bert-base-arabic-camelbert-mix"


class GeneralityDataset(Dataset):
    def __init__(self, texts, scores, tokenizer, max_length=512):
        self.texts = list(texts)
        self.scores = list(scores)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.scores[idx], dtype=torch.float)
        return item


class BertRegressor(nn.Module):
    """Exp_2 cell 46: pooled [CLS] -> dropout -> Linear(1)."""

    def __init__(self, model_name):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = self.dropout(outputs.pooler_output)
        prediction = self.regressor(x).squeeze(-1)
        if labels is not None:
            return nn.MSELoss()(prediction, labels), prediction
        return prediction


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _evaluate(model, loader, device, loss_fn):
    model.eval()
    losses, preds, labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)
            out = model(input_ids=ids, attention_mask=mask)
            losses.append(loss_fn(out, y).item())
            preds.extend(out.cpu().numpy())
            labels.extend(y.cpu().numpy())
    model.train()
    return float(np.mean(losses)), float(np.sqrt(mean_squared_error(labels, preds)))


def train(args):
    _seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    if not os.path.isdir(args.tokenizer_dir):
        logger.info("Tokenizer dir missing; building one at %s", args.tokenizer_dir)
        prepare_tokenizer("sentence", args.base_model, args.tokenizer_dir)
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_dir)

    import pandas as pd
    train_df = pd.read_csv(args.train_csv)
    dev_df = pd.read_csv(args.dev_csv)
    train_ds = GeneralityDataset(train_df["sentence"], train_df["score"], tokenizer, args.max_length)
    dev_ds = GeneralityDataset(dev_df["sentence"], dev_df["score"], tokenizer, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size)

    model = BertRegressor(args.base_model)
    model.bert.resize_token_embeddings(len(tokenizer))
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    num_training_steps = len(train_loader) * args.epochs
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0,
                                 num_training_steps=num_training_steps)
    loss_fn = nn.MSELoss()

    if args.fresh and os.path.isdir(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    dev_log_path = os.path.join(args.output_dir, "dev_logs.csv")
    with open(dev_log_path, "w") as f:
        f.write("step,dev_loss,dev_rmse\n")

    wb = None
    if args.wandb_project:
        import wandb
        wb = wandb.init(project=args.wandb_project, name=args.wandb_run, resume="allow")

    global_step = 0
    stop = False
    for epoch in range(args.epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            global_step += 1
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)

            out = model(input_ids=ids, attention_mask=mask)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if wb:
                wb.log({"train_loss": loss.item()}, step=global_step)

            if global_step % args.checkpoint_steps == 0:
                ckpt = os.path.join(args.output_dir, f"checkpoint_step{global_step}.pt")
                torch.save(model.state_dict(), ckpt)
                dev_loss, dev_rmse = _evaluate(model, dev_loader, device, loss_fn)
                with open(dev_log_path, "a") as f:
                    f.write(f"{global_step},{dev_loss:.6f},{dev_rmse:.6f}\n")
                logger.info("step %d | dev_loss %.4f | dev_rmse %.4f -> %s",
                            global_step, dev_loss, dev_rmse, ckpt)
                if wb:
                    wb.log({"dev_loss": dev_loss, "dev_rmse": dev_rmse}, step=global_step)

            if args.max_steps and global_step >= args.max_steps:
                stop = True
                break
        if stop:
            break

    final = os.path.join(args.output_dir, "final_model.pt")
    torch.save(model.state_dict(), final)
    logger.info("Final model -> %s", final)
    if wb:
        wb.finish()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-csv", required=True)
    p.add_argument("--dev-csv", required=True)
    p.add_argument("--tokenizer-dir", default="models/ags_tokenizer")
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument("--output-dir", default="models/ags_sentence")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=4e-5)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--checkpoint-steps", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=None, help="Stop early (smoke tests)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fresh", action="store_true", help="Delete --output-dir before training (Exp_2 default)")
    p.add_argument("--wandb-project", default=None, help="Enable wandb logging under this project")
    p.add_argument("--wandb-run", default=None)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
