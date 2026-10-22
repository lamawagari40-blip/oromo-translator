"""Gradio web UI for Oromo translation."""
import gradio as gr
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_DIR = {
    "om_to_am": "lwagari/nllb-om_to_am",
    "am_to_om": "lwagari/nllb-am_to_om"
}
LANG_CODE = {"om": "gaz_Latn", "am": "amh_Ethi", "en": "eng_Latn"}
_cache = {}


def load(direction):
    if direction not in _cache:
        from transformers import BitsAndBytesConfig
        path = MODEL_DIR.get(direction)
        if not path:
            raise FileNotFoundError(f"No model: {direction}")
            
        # Load tokenizer from HF
        tok = AutoTokenizer.from_pretrained(path)
        
        # Load model in int8 to fit in Streamlit's 1GB RAM limit
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            path, 
            quantization_config=bnb_config,
            device_map="auto"
        ).eval()
        _cache[direction] = (tok, model)
    return _cache[direction]


def translate(text, src, tgt):
    if not text or not text.strip():
        return ""
    if src == tgt:
        return text
    direction = f"{src}_to_{tgt}"
    try:
        tok, model = load(direction)
    except FileNotFoundError:
        return f"[Model not available: {src}→{tgt}]"
    tok.src_lang = LANG_CODE[src]
    enc = tok(text, return_tensors="pt", truncation=True, max_length=128)
    if torch.cuda.is_available():
        enc = {k: v.cuda() for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(
            **enc,
            forced_bos_token_id=tok.convert_tokens_to_ids(LANG_CODE[tgt]),
            max_length=128, num_beams=4, early_stopping=True,
        )
    return tok.decode(out[0], skip_special_tokens=True)


def swap(src, tgt):
    return tgt, src


with gr.Blocks(title="Oromo Translator") as demo:
    gr.Markdown("# 🌍 Afaan Oromoo ↔ Amharic ↔ English")
    gr.Markdown("Bidirectional Machine Translation Prototype")

    with gr.Row():
        src_dd = gr.Dropdown(["om", "am", "en"], value="om", label="From")
        swap_btn = gr.Button("⇄", scale=0)
        tgt_dd = gr.Dropdown(["om", "am", "en"], value="en", label="To")

    input_box = gr.Textbox(label="Input", placeholder="Akkam nagaa dha?", lines=3)
    translate_btn = gr.Button("Translate", variant="primary")
    output_box = gr.Textbox(label="Translation", lines=3)

    translate_btn.click(translate, [input_box, src_dd, tgt_dd], output_box)
    input_box.submit(translate, [input_box, src_dd, tgt_dd], output_box)
    swap_btn.click(swap, [src_dd, tgt_dd], [src_dd, tgt_dd])

    gr.Examples(
        examples=[
            ["Akkam nagaa dha?", "om", "am"],
            ["Galatoomi.", "om", "am"],
            ["ሰላም ነህ?", "am", "om"],
        ],
        inputs=[input_box, src_dd, tgt_dd],
    )

if __name__ == "__main__":
    demo.launch(share=True, theme=gr.themes.Soft())
