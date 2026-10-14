import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

USERNAME = "lwagari"
LANG_CODE = {"om": "gaz_Latn", "am": "amh_Ethi", "en": "eng_Latn"}
LANG_NAME = {"om": "Afaan Oromoo", "am": "Amharic", "en": "English"}

st.set_page_config(page_title="Oromo Translator", page_icon="🌍")
st.title("🌍 Afaan Oromoo ↔ Amharic ↔ English")
st.caption("Bidirectional Machine Translation Prototype")

@st.cache_resource
def load(direction):
    repo_id = f"{USERNAME}/nllb-{direction}"
    tok = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(repo_id)
    model.eval()
    return tok, model

def translate(text, src, tgt):
    if not text.strip() or src == tgt:
        return text
    try:
        tok, model = load(f"{src}_to_{tgt}")
    except Exception as e:
        return f"[Model not available: {src}→{tgt}]"
    tok.src_lang = LANG_CODE[src]
    enc = tok(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        out = model.generate(
            **enc,
            forced_bos_token_id=tok.convert_tokens_to_ids(LANG_CODE[tgt]),
            max_length=128, num_beams=4,
        )
    return tok.decode(out[0], skip_special_tokens=True)

col1, col2 = st.columns(2)
src = col1.selectbox("From", ["om", "am"], format_func=lambda x: LANG_NAME[x])
tgt = col2.selectbox("To", ["am", "om"], format_func=lambda x: LANG_NAME[x])

text = st.text_area("Input text", "Akkam nagaa dha?")

if st.button("Translate", type="primary"):
    result = translate(text, src, tgt)
    st.text_area("Translation", result, height=100)
