"""Test translation with trained NLLB model."""
import sys
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_DIR = Path("artifacts/models")
LANG_CODE = {"om": "gaz_Latn", "am": "amh_Ethi", "en": "eng_Latn"}


def translate(text, src, tgt):
    direction = f"{src}_to_{tgt}"
    model_path = MODEL_DIR / direction
    if not model_path.exists():
        return f"Model not found: {direction}"

    tok = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path))
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    tok.src_lang = LANG_CODE[src]
    enc = tok(text, return_tensors="pt", truncation=True, max_length=128)
    if torch.cuda.is_available():
        enc = {k: v.cuda() for k, v in enc.items()}

    with torch.no_grad():
        out = model.generate(
            **enc,
            forced_bos_token_id=tok.convert_tokens_to_ids(LANG_CODE[tgt]),
            max_length=128,
            num_beams=4,
        )
    return tok.decode(out[0], skip_special_tokens=True)


if __name__ == "__main__":
    # Test sentences
    tests = [
        ("Akkam nagaa dha?", "om", "am"),
        ("Galatoomi.", "om", "am"),
        ("Ani barataa dha.", "om", "am"),
        ("Bishaan kootu guddaa dha.", "om", "am"),
    ]
    print("=" * 60)
    print("TRANSLATION TEST (om -> am)")
    print("=" * 60)
    for text, src, tgt in tests:
        result = translate(text, src, tgt)
        print(f"\n{src.upper()}: {text}")
        print(f"{tgt.upper()}: {result}")
    print()
