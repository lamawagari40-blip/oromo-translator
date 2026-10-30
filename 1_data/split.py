"""Split cleaned data into train/val/test (80/10/10)."""
import pandas as pd
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

CLEAN_DIR = Path("artifacts/data/cleaned")
SPLIT_DIR = Path("artifacts/data/splits")
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

def split_one(csv_file, src_col, tgt_col, pair_name, seed=42):
    print(f"\n{'='*60}")
    print(f"Splitting {csv_file} -> {pair_name}")
    print(f"{'='*60}")

    df = pd.read_csv(CLEAN_DIR / csv_file)
    print(f"Total: {len(df)}")

    train, temp = train_test_split(df, test_size=0.2, random_state=seed)
    val, test = train_test_split(temp, test_size=0.5, random_state=seed)

    out = SPLIT_DIR / pair_name
    out.mkdir(parents=True, exist_ok=True)

    for name, data in [("train", train), ("val", val), ("test", test)]:
        data[src_col].to_csv(out / f"{name}.{src_col}", index=False, header=False)
        data[tgt_col].to_csv(out / f"{name}.{tgt_col}", index=False, header=False)
        print(f"  {name}: {len(data)}")

    stats = {
        "pair": pair_name,
        "src_lang": src_col,
        "tgt_lang": tgt_col,
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "seed": seed
    }
    with open(out / "split_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    return stats

def run():
    all_stats = []
    # Oromo-English
    all_stats.append(split_one("om_en_clean.csv", "om", "en", "om_en"))
    # Amharic-English
    all_stats.append(split_one("am_en_clean.csv", "am", "en", "am_en"))
    # Oromo-Amharic
    all_stats.append(split_one("om_am_clean.csv", "om", "am", "om_am"))

    with open(SPLIT_DIR / "all_splits.json", "w") as f:
        json.dump(all_stats, f, indent=2)
    print("\nAll splits complete.")

if __name__ == "__main__":
    run()
