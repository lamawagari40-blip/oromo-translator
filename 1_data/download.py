"""Download parallel corpora from HuggingFace."""
import os
from pathlib import Path
from datasets import load_dataset
import pandas as pd

RAW_DIR = Path("artifacts/data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

def save_hf_dataset(dataset_name, src_col, tgt_col, out_csv, out_src, out_tgt):
    """Load HuggingFace dataset and save to CSV."""
    print(f"Loading {dataset_name}...")
    try:
        ds = load_dataset(dataset_name, split="train")
        df = ds.to_pandas()
        df = df[[src_col, tgt_col]].dropna()
        df.columns = [out_src, out_tgt]
        df.to_csv(RAW_DIR / out_csv, index=False, encoding="utf-8-sig")
        print(f"  DONE {out_csv}: {len(df)} rows")
    except Exception as e:
        print(f"  FAILED {dataset_name}: {e}")

def run():
    # Oromo-English
    save_hf_dataset(
        "michsethowusu/english-oromo_sentence-pairs_mt560",
        "orm", "eng",
        "om_en.csv", "om", "en"
    )
    # Amharic-English
    save_hf_dataset(
        "michsethowusu/english-amharic_sentence-pairs_mt560",
        "amh", "eng",
        "am_en.csv", "am", "en"
    )
    print("\nDownloads complete. Check artifacts/data/raw/")

if __name__ == "__main__":
    run()
