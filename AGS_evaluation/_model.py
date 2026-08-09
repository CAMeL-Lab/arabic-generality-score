"""Shared inference helpers for the ``.pt`` BertRegressor checkpoints from AGS_training."""

import torch
from transformers import AutoTokenizer

from AGS_training.train_sentence_ags import BASE_MODEL, BertRegressor
from utilities.preprocess_text import preprocess_text

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_tokenizer(tokenizer_dir: str):
    return AutoTokenizer.from_pretrained(tokenizer_dir)


def load_gen_model(checkpoint_path: str, tokenizer, base_model: str = BASE_MODEL) -> BertRegressor:
    """Build the module, load a raw ``state_dict`` checkpoint, put it in eval mode."""
    model = BertRegressor(base_model)
    model.bert.resize_token_embeddings(len(tokenizer))
    model.to(DEVICE)
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def mark_target_word(sentence: str, target_word: str):
    if target_word in sentence:
        return sentence.replace(target_word, f"[TGT]{target_word}[/TGT]", 1)
    return None


@torch.inference_mode()
def predict_score(model, tokenizer, sentence: str, target_word: str, max_length: int = 128) -> float:
    marked = mark_target_word(sentence, target_word)
    if marked is None:
        raise ValueError("Target word not found in the sentence.")
    enc = tokenizer(marked, padding="max_length", truncation=True,
                    max_length=max_length, return_tensors="pt")
    ids = enc["input_ids"].to(DEVICE)
    mask = enc["attention_mask"].to(DEVICE)
    return float(model(ids, attention_mask=mask).item())


def predict_scores_sent(model, tokenizer, sentence: str):
    """Per-word AGS for a sentence (underscores treated as spaces, then normalized)."""
    sentence = " ".join(sentence.split("_"))
    sentence = preprocess_text(sentence)
    return [predict_score(model, tokenizer, sentence, w) for w in sentence.split()]


def hmean(word_scores, eps: float = 1e-6) -> float:
    """Harmonic mean of a list of scores."""
    safe = [max(s, eps) for s in word_scores]
    denom = sum(1.0 / s for s in safe)
    return len(safe) / denom if denom else 0.0


def sentence_generality(word_scores, k: int) -> float:
    """Sentence-level generality: harmonic mean of the k lowest word scores."""
    return hmean(sorted(word_scores)[:k])
