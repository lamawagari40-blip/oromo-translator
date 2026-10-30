"""Fine-tune NLLB-200 for Oromo/Amharic/English MT."""
import argparse
import json
import torch
from pathlib import Path
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

SPLIT_DIR = Path("artifacts/data/splits")
CKPT_DIR = Path("artifacts/checkpoints")
MODEL_DIR = Path("artifacts/models")

MODEL_NAME = "facebook/nllb-200-distilled-600M"

LANG_CODE = {
    "om": "gaz_Latn",
    "am": "amh_Ethi",
    "en": "eng_Latn",
}


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
        # Source encoding - set source language
        self.tok.src_lang = LANG_CODE[self.src_lang]
        enc = self.tok(
            self.src[idx],
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
        )
        # Target encoding - use text_target parameter (new API)
        self.tok.src_lang = LANG_CODE[self.tgt_lang]
        labels = self.tok(
            text_target=self.tgt[idx],
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
        )
        enc["labels"] = labels["input_ids"]
        # Force target language token for generation
        enc["forced_bos_token_id"] = self.tok.convert_tokens_to_ids(
            LANG_CODE[self.tgt_lang]
        )
        return enc


def train(direction, epochs, batch_size, lr):
    src_lang, tgt_lang = direction.split("_to_")
    pair_name = f"{src_lang}_{tgt_lang}" if f"{src_lang}_{tgt_lang}" in [
        "om_en", "am_en", "om_am"
    ] else f"{tgt_lang}_{src_lang}"

    pair_dir = SPLIT_DIR / pair_name
    print(f"\n{'='*60}")
    print(f"Training: {direction}")
    print(f"Pair:     {pair_name}")
    print(f"Data:     {pair_dir}")
    print(f"{'='*60}")

    if not pair_dir.exists():
        print(f"ERROR: {pair_dir} not found")
        return

    print(f"Loading {MODEL_NAME}...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    train_ds = ParallelDataset(pair_dir, "train", tok, src_lang, tgt_lang)
    val_ds = ParallelDataset(pair_dir, "val", tok, src_lang, tgt_lang)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    out_dir = CKPT_DIR / direction
    out_dir.mkdir(parents=True, exist_ok=True)

    args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        warmup_steps=500,
        save_steps=2000,
        eval_steps=2000,
        logging_steps=50,
        save_total_limit=2,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
        generation_max_length=128,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tok, model=model),
    )

    print("Starting training...")
    trainer.train()

    final = MODEL_DIR / direction
    final.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final)
    tok.save_pretrained(final)
    print(f"Saved: {final}")


def run():
    p = argparse.ArgumentParser()
    p.add_argument("--direction", required=True,
                   choices=["om_to_en", "en_to_om", "am_to_en",
                            "en_to_am", "om_to_am", "am_to_om"])
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-5)
    a = p.parse_args()
    train(a.direction, a.epochs, a.batch_size, a.lr)


if __name__ == "__main__":
    run()
