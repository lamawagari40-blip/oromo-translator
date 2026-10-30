"""16-step cleaning pipeline for parallel corpora."""
import pandas as pd
import re
import ftfy
import unicodedata
import json
from pathlib import Path
from tqdm import tqdm

RAW_DIR = Path("artifacts/data/raw")
CLEAN_DIR = Path("artifacts/data/cleaned")
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

tqdm.pandas()

# ==========================================
# TEXT CLEANING FUNCTIONS
# ==========================================
def normalize_unicode(text):
    if not isinstance(text, str): return ""
    text = ftfy.fix_text(text)
    return unicodedata.normalize("NFC", text)

def clean_whitespace(text):
    text = re.sub(r"[\u200b\u200c\u200d\xa0\t\n\r]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def normalize_punct(text):
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("—", "-").replace("–", "-").replace("…", "...")
    return text

def remove_noise(text):
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"[\U0001F600-\U0001F64F]", "", text)
    return text.strip()

def clean_text(text):
    text = normalize_unicode(text)
    text = clean_whitespace(text)
    text = normalize_punct(text)
    text = remove_noise(text)
    return clean_whitespace(text)

def is_valid(src, tgt, min_len=1, max_len=128, min_r=0.3, max_r=3.0):
    if not src or not tgt: return False
    s, t = src.split(), tgt.split()
    if not (min_len <= len(s) <= max_len): return False
    if not (min_len <= len(t) <= max_len): return False
    r = len(s) / max(len(t), 1)
    return min_r <= r <= max_r

# ==========================================
# CLEAN ONE FILE
# ==========================================
def clean_pair(csv_file, src_col, tgt_col):
    print(f"\n{'='*60}")
    print(f"Cleaning {csv_file}")
    print(f"{'='*60}")
    df = pd.read_csv(RAW_DIR / csv_file)
    start = len(df)
    print(f"Loaded: {start}")

    df = df.dropna(subset=[src_col, tgt_col])
    df = df[df[src_col].astype(str).str.strip() != ""]
    df = df[df[tgt_col].astype(str).str.strip() != ""]
    after_missing = len(df)
    print(f"After drop missing: {after_missing}")

    print("Cleaning source...")
    df[src_col] = df[src_col].progress_apply(clean_text)
    print("Cleaning target...")
    df[tgt_col] = df[tgt_col].progress_apply(clean_text)

    before = len(df)
    df = df.drop_duplicates(subset=[src_col, tgt_col])
    dup_dropped = before - len(df)
    print(f"Duplicates removed: {dup_dropped}")

    before = len(df)
    df = df[df.progress_apply(lambda r: is_valid(r[src_col], r[tgt_col]), axis=1)]
    len_dropped = before - len(df)
    print(f"Length/ratio filter removed: {len_dropped}")

    out = CLEAN_DIR / csv_file.replace(".csv", "_clean.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")

    report = {
        "file": csv_file,
        "input": start,
        "output": len(df),
        "dropped_missing": start - after_missing,
        "dropped_duplicates": dup_dropped,
        "dropped_length_ratio": len_dropped,
        "retention_pct": round(100 * len(df) / start, 2)
    }
    print(f"Saved {out.name}: {len(df)} rows ({report['retention_pct']}%)")
    return report

# ==========================================
# RUN ALL
# ==========================================
def run():
    reports = []
    reports.append(clean_pair("om_en.csv", "om", "en"))
    reports.append(clean_pair("am_en.csv", "am", "en"))
    reports.append(clean_pair("om_am.csv", "om", "am"))

    with open(CLEAN_DIR / "cleaning_report.json", "w") as f:
        json.dump(reports, f, indent=2)
    print("\nCleaning complete.")

if __name__ == "__main__":
    run()
