"""Create om_am.csv from HornMT text files."""
import pandas as pd
from pathlib import Path

HORNMT_DIR = Path("/content/HornMT/data")
RAW_DIR = Path("artifacts/data/raw")

def create_om_am():
    om_file = HORNMT_DIR / "orm.txt"
    am_file = HORNMT_DIR / "amh.txt"

    if not om_file.exists() or not am_file.exists():
        print(f"Error: Could not find orm.txt or amh.txt in {HORNMT_DIR}")
        return

    # Read the files
    with open(om_file, encoding="utf-8") as f:
        om_lines = [line.strip() for line in f]

    with open(am_file, encoding="utf-8") as f:
        am_lines = [line.strip() for line in f]

    # Ensure they are the same length
    if len(om_lines) != len(am_lines):
        print(f"Error: Line count mismatch. Oromo: {len(om_lines)}, Amharic: {len(am_lines)}")
        return

    # Create DataFrame
    df = pd.DataFrame({"om": om_lines, "am": am_lines})
    df = df.dropna().drop_duplicates()

    # Save to CSV
    output_path = RAW_DIR / "om_am.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Successfully created {output_path} with {len(df)} rows.")

if __name__ == "__main__":
    create_om_am()
