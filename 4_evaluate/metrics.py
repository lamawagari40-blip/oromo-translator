"""Compute BLEU, chrF, TER for trained models."""
import json, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sacrebleu.metrics import BLEU, CHRF, TER
from tqdm import tqdm

SPLIT_DIR = Path("artifacts/data/splits")
MODEL_DIR = Path("artifacts/models")
REPORT_DIR = Path("artifacts/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

LANG_CODE = {"om": "gaz_Latn", "am": "amh_Ethi", "en": "eng_Latn"}


def evaluate_direction(src, tgt, pair, max_n=300):
    model_path = MODEL_DIR / f"{src}_to_{tgt}"
    if not model_path.exists():
        print(f"SKIP {src}_to_{tgt}: model not found")
        return None

    print(f"\n{'='*60}\nEvaluating {src} -> {tgt}\n{'='*60}")
    tok = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path)).eval()
    if torch.cuda.is_available():
        model = model.cuda()

    src_file = SPLIT_DIR / pair / f"test.{src}"
    tgt_file = SPLIT_DIR / pair / f"test.{tgt}"
    srcs = open(src_file, encoding="utf-8").read().splitlines()[:max_n]
    refs = open(tgt_file, encoding="utf-8").read().splitlines()[:max_n]
    print(f"Test samples: {len(srcs)}")

    preds = []
    for i in tqdm(range(0, len(srcs), 16), desc=f"{src}->{tgt}"):
        batch = srcs[i:i+16]
        tok.src_lang = LANG_CODE[src]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
        if torch.cuda.is_available():
            enc = {k: v.cuda() for k, v in enc.items()}
        with torch.no_grad():
            out = model.generate(**enc,
                forced_bos_token_id=tok.convert_tokens_to_ids(LANG_CODE[tgt]),
                max_length=128, num_beams=4)
        preds.extend(tok.batch_decode(out, skip_special_tokens=True))

    bleu = BLEU().corpus_score(preds, [refs]).score
    chrf = CHRF().corpus_score(preds, [refs]).score
    ter = TER().corpus_score(preds, [refs]).score

    print(f"BLEU: {bleu:.2f}")
    print(f"chrF: {chrf:.2f}")
    print(f"TER:  {ter:.2f}")

    report = {"BLEU": round(bleu, 2), "chrF": round(chrf, 2),
              "TER": round(ter, 2), "n": len(preds)}
    with open(REPORT_DIR / f"metrics_{src}_to_{tgt}.json", "w") as f:
        json.dump(report, f, indent=2)

    # Save predictions for error analysis
    with open(REPORT_DIR / f"preds_{src}_to_{tgt}.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(preds))
    return report


def run():
    all_res = {}
    directions = [
        ("om", "am", "om_am"),
        ("am", "om", "om_am"),
    ]
    for src, tgt, pair in directions:
        r = evaluate_direction(src, tgt, pair)
        if r:
            all_res[f"{src}_to_{tgt}"] = r

    with open(REPORT_DIR / "all_metrics.json", "w") as f:
        json.dump(all_res, f, indent=2)

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"{'Direction':<15} {'BLEU':>8} {'chrF':>8} {'TER':>8}")
    print("-" * 45)
    for d, r in all_res.items():
        print(f"{d:<15} {r['BLEU']:>8.2f} {r['chrF']:>8.2f} {r['TER']:>8.2f}")


if __name__ == "__main__":
    run()
