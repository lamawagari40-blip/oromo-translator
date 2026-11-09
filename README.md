# 🌳 Saiko — Trilingual Machine Translation

**Afaan Oromoo ↔ Amharic ↔ English**

A from-scratch PyTorch Neural Machine Translation (NMT) system built for a Computer Engineering Internship at **Jimma University** and **EAII**.

## 🔗 Live Links
- 🌐 **Live App:** [saikoai.streamlit.app](https://saikoai.streamlit.app/)
- 🤖 **AI Model:** [huggingface.co/lwagari/saiko-lstm-model](https://huggingface.co/lwagari/saiko-lstm-model)
- 📊 **Dataset:** [huggingface.co/datasets/lwagari/saiko-data](https://huggingface.co/datasets/lwagari/saiko-data)

## 🏗️ Architecture
- **Type:** Sequence-to-Sequence (Seq2Seq)
- **Encoder:** 2-Layer Bidirectional LSTM
- **Attention:** Bahdanau Attention (additive)
- **Decoder:** 2-Layer Unidirectional LSTM
- **Vocabulary:** 50,004 custom tokens
- **Parameters:** ~81.5 Million (built from zero)

## 📊 Dataset Engineering
- **1,706,118** trilingual pairs synthesized via the **English Pivot Strategy**.
- Sources: MT560 (HuggingFace), HornMT (news), and 1,000+ oversampled conversational phrases.
- 16-step cleaning pipeline with length-ratio filtering.

## 🚀 Quick Start
```bash
git clone https://github.com/lamawagari40-blip/oromo-translator
cd oromo-translator
pip install -r requirements.txt
streamlit run streamlit_app.py
