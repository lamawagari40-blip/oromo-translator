"""Train on a sample of data for quick iteration."""
import argparse
import random
import torch
from pathlib import Path
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq,
)

SPLIT_DIR = Path("artifacts/data/splits")
SAMPLE_DIR = Path("artifacts/data/splits_sampled")
CKPT_DIR = Path("artifacts/checkpoints")
MODEL_DIR = Path("artifacts/models")
MODEL_NAME = "facebook/nllb-200-distilled-600M"
LANG_CODE = {"om": "gaz_Latn", "am": "amh_Ethi", "en": "eng_Latn"}


class ParallelDataset(Dataset):
    def __init__(self, pair_dir, split, tokenizer, src_lang, tgt_lang, max_len=128):
        self.tok = tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_len = max_len
        self.src_file = Path(pair_dir) / f"{split}.{src_lang}"
        self.tgt_file = Path(pair_dir) / f"{split}.{tgt_lang}"
        with open(self.src_file, encoding="utf-8") as f:
            self.src = [line.strip() for line in f]
        with open(self.tgt_file, encoding="utf-8") as f:
            self.tgt = [line.strip() for line in f]

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        self.tok.src_lang = LANG_CODE[self.src_lang]
        enc = self.tok(self.src[idx], max_length=self.max_len,
                       truncation=True, padding="max_length")
        self.tok.src_lang = LANG_CODE[self.tgt_lang]
        labels = self.tok(text_target=self.tgt[idx], max_length=self.max_len,
                          truncation=True, padding="max_length")
        enc["labels"] = labels["input_ids"]
        enc["forced_bos_token_id"] = self.tok.convert_tokens_to_ids(
            LANG_CODE[self.tgt_lang])
        return enc


def sample_data(src_file, tgt_file, out_dir, n):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src_lines = open(src_file, encoding="utf-8").read().splitlines()
    tgt_lines = open(tgt_file, encoding="utf-8").read().splitlines()
    assert len(src_lines) == len(tgt_lines), "Length mismatch"
    pairs = list(zip(src_lines, tgt_lines))
    random.seed(42)
    random.shuffle(pairs)
    sample = pairs[:n]
    with open(out_dir / src_file.name, "w", encoding="utf-8") as f:
        for s, _ in sample: f.write(s + "\n")
    with open(out_dir / tgt_file.name, "w", encoding="utf-8") as f:
        for _, t in sample: f.write(t + "\n")
    print(f"Sampled {len(sample)} -> {out_dir}")
    return len(sample)


def prepare_samples(pair_name, src_lang, tgt_lang, n_train=20000, n_val=1000):
    src_dir = SPLIT_DIR / pair_name
    out_dir = SAMPLE_DIR / pair_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Sampling from {pair_name}...")
    sample_data(src_dir / f"train.{src_lang}", src_dir / f"train.{tgt_lang}",
                out_dir, n_train)
    sample_data(src_dir / f"val.{src_lang}", src_dir / f"val.{tgt_lang}",
                out_dir, n_val)


def train_sampled(direction, epochs, batch_size, lr):
    src_lang, tgt_lang = direction.split("_to_")
    pair_name = f"{src_lang}_{tgt_lang}" if f"{src_lang}_{tgt_lang}" in [
        "om_en", "am_en", "om_am"
    ] else f"{tgt_lang}_{src_lang}"

    prepare_samples(pair_name, src_lang, tgt_lang)
    sample_dir = SAMPLE_DIR / pair_name

    print(f"\n{'='*60}\nTraining: {direction} (SAMPLED)\n{'='*60}")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    train_ds = ParallelDataset(sample_dir, "train", tok, src_lang, tgt_lang)
    val_ds = ParallelDataset(sample_dir, "val", tok, src_lang, tgt_lang)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    out_dir = CKPT_DIR / f"{direction}_sampled"
    out_dir.mkdir(parents=True, exist_ok=True)

    args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        warmup_steps=200,
        save_steps=2000,
        logging_steps=100,
        save_total_limit=1,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
    trainer = Seq2SeqTrainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tok, model=model),
    )
    print("Starting training...")
    trainer.train()

    final = MODEL_DIR / direction
    final.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final)
    tok.save_pretrained(final)
    print(f"Saved: {final}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--direction", required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-5)
    a = p.parse_args()
    train_sampled(a.direction, a.epochs, a.batch_size, a.lr)
