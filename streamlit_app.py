import streamlit as st
from huggingface_hub import InferenceClient

# Read token from Streamlit secrets (set in Streamlit Cloud dashboard)
try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except Exception:
    HF_TOKEN = None

MODEL_ID = "facebook/nllb-200-distilled-600M"

if not HF_TOKEN:
    st.error("Missing HF_TOKEN in Streamlit secrets. Add it in app settings.")
    st.stop()

client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)

LANG_CODE = {"om": "gaz_Latn", "am": "amh_Ethi", "en": "eng_Latn"}
LANG_NAME = {"om": "Afaan Oromoo", "am": "Amharic", "en": "English"}

st.set_page_config(page_title="Oromo Translator", page_icon="🌍")
st.title("🌍 Afaan Oromoo ↔ Amharic ↔ English")
st.caption("Bidirectional Machine Translation Prototype")

col1, col2 = st.columns(2)
src = col1.selectbox("From", ["om", "am"], format_func=lambda x: LANG_NAME[x])
tgt = col2.selectbox("To", ["am", "om"], format_func=lambda x: LANG_NAME[x])

text = st.text_area("Input text", "Akkam nagaa dha?")

if st.button("Translate", type="primary"):
    if not text.strip():
        st.warning("Please enter some text.")
    elif src == tgt:
        st.info(text)
    else:
        with st.spinner("Translating..."):
            try:
                response = client.translation(
                    text,
                    src_lang=LANG_CODE[src],
                    tgt_lang=LANG_CODE[tgt],
                )
                result = response["translation_text"]
                st.text_area("Translation", result, height=100)
            except Exception as e:
                st.error(f"API Error: {e}")
                st.info("The free API may be busy. Please try again.")
