import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
import torch

LANG_NAME = {"om": "Afaan Oromoo", "am": "Amharic"}
LANG_CODE = {"om": "gaz_Latn", "am": "amh_Ethi"}
MODELS = {
    "om_to_am": "lwagari/nllb-om_to_am",
    "am_to_om": "lwagari/nllb-am_to_om",
}

@st.cache_resource(show_spinner="Loading model (first time only, ~30s)...")
def load_model(direction):
    repo_id = MODELS[direction]
    tok = AutoTokenizer.from_pretrained(repo_id)
    bnb = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        repo_id,
        quantization_config=bnb,
        device_map="auto",
    )
    model.eval()
    return tok, model

def translate(text, src, tgt):
    if not text.strip() or src == tgt:
        return text
    tok, model = load_model(f"{src}_to_{tgt}")
    tok.src_lang = LANG_CODE[src]
    enc = tok(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        out = model.generate(
            **enc,
            forced_bos_token_id=tok.convert_tokens_to_ids(LANG_CODE[tgt]),
            max_length=128,
            num_beams=4,
        )
    return tok.decode(out[0], skip_special_tokens=True)

st.set_page_config(page_title="Oromo Translator", page_icon="🌍")
st.title("🌍 Afaan Oromoo ↔ Amharic")
st.caption("Bidirectional Machine Translation Prototype")

col1, col2 = st.columns(2)
src = col1.selectbox("From", ["om", "am"], format_func=lambda x: LANG_NAME[x])
tgt = col2.selectbox("To", ["am", "om"], format_func=lambda x: LANG_NAME[x])

text = st.text_area("Input text", "Akkam nagaa dha?")

if st.button("Translate", type="primary"):
    with st.spinner("Translating..."):
        try:
            result = translate(text, src, tgt)
            st.text_area("Translation", result, height=100)
        except Exception as e:
            st.error(f"Error: {e}")
