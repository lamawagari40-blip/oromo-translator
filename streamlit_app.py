
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Saiko", page_icon="🌳", layout="centered")


st.markdown('''
    <style>
    /* Hide Streamlit Header and Footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Force the main background color */
    .stApp {
        background-color: #0E1117 !important;
    }
    
    /* Remove the top padding so Saiko is higher up */
    .block-container {
        padding-top: 2rem !important;
    }
    </style>
''', unsafe_allow_html=True)


st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.markdown('''
    <style>
    .title-text {
        font-size: 3.5rem !important;
        font-weight: 900 !important;
        background: -webkit-linear-gradient(45deg, #006633, #FFCC00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0px;
    }
    .welcome-text {
        text-align: center;
        font-size: 1.4rem;
        font-weight: 600;
        color: #CE1126;
        margin-bottom: 5px;
    }
    .subtitle-text {
        text-align: center;
        font-size: 1rem;
        color: #A0AAB2;
        margin-bottom: 40px;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #006633 0%, #FFCC00 50%, #CE1126 100%);
        color: white;
        font-weight: 800;
        font-size: 1.2rem;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0px 6px 20px rgba(255, 204, 0, 0.4);
    }
    </style>
''', unsafe_allow_html=True)

st.markdown('<h1 class="title-text">🌳 Saiko</h1>', unsafe_allow_html=True)
st.markdown('<p class="welcome-text">Baga Nagaan Dhuftan!</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Trilingual Machine Translation</p>', unsafe_allow_html=True)

device = torch.device("cpu")

class Vocab:
    def __init__(self, sentences=None, min_freq=10):
        self.word2id = {}
        self.id2word = {}
    def encode(self, text):
        tokens = [self.word2id.get(w, self.word2id.get("<unk>", 3)) for w in str(text).strip().split()]
        return [self.word2id.get("<sos>", 1)] + tokens + [self.word2id.get("<eos>", 2)]

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, enc_hid_dim, num_layers=2, bidirectional=True, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)
    def forward(self, hidden, encoder_outputs):
        batch_size, src_len = encoder_outputs.shape[0], encoder_outputs.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM((enc_hid_dim * 2) + emb_dim, dec_hid_dim, num_layers=2, batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))
        a = self.attention(hidden[-1], encoder_outputs).unsqueeze(1)
        weighted = torch.bmm(a, encoder_outputs) 
        rnn_input = torch.cat((weighted, embedded), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        prediction = self.fc_out(torch.cat((output.squeeze(1), weighted.squeeze(1), embedded.squeeze(1)), dim=1))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

@st.cache_resource(show_spinner="Loading Saiko AI from Hugging Face...")
def load_model():
    vocab_path = hf_hub_download(repo_id="lwagari/saiko-lstm-model", filename="vocab.pkl")
    model_path = hf_hub_download(repo_id="lwagari/saiko-lstm-model", filename="lstm_focus.pt")

    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    
    INPUT_DIM = OUTPUT_DIM = len(vocab.word2id)
    attn = Attention(256, 256)
    enc = Encoder(INPUT_DIM, 256, 256, 256, 0.1)
    dec = Decoder(OUTPUT_DIM, 256, 256, 256, 0.1, attn)
    model = Seq2Seq(enc, dec, device).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, vocab

def translate(model, vocab, sentence, tgt_lang="am"):
    reversed_words = list(reversed(sentence.split()))
    src_text = f"<2{tgt_lang}> " + " ".join(reversed_words)
    
    src_tensor = torch.tensor(vocab.encode(src_text)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)
        hidden = hidden.unsqueeze(0).repeat(2, 1, 1)
        cell = torch.zeros_like(hidden).to(device)
        
        # BEAM SEARCH IMPLEMENTATION
        k = 3 # Beam width (looks at top 3 paths)
        max_len = 20
        
        # Beam contains tuples of (log_probability, sequence, hidden, cell)
        beam = [(0.0, [vocab.word2id["<sos>"]], hidden, cell)]
        
        for _ in range(max_len):
            candidates = []
            for score, seq, h, c in beam:
                # If the sequence already ended, keep it
                if seq[-1] == vocab.word2id["<eos>"]:
                    candidates.append((score, seq, h, c))
                    continue
                
                trg_tensor = torch.tensor([seq[-1]]).to(device)
                output, new_h, new_c = model.decoder(trg_tensor, h, c, encoder_outputs)
                
                # Get top k tokens
                log_probs = F.log_softmax(output, dim=1)
                topk_scores, topk_ids = log_probs.topk(k, dim=1)
                
                for i in range(k):
                    token = topk_ids[0][i].item()
                    new_score = score + topk_scores[0][i].item()
                    new_seq = seq + [token]
                    candidates.append((new_score, new_seq, new_h, new_c))
            
            # Sort candidates by score and keep the top k
            candidates.sort(key=lambda x: x[0], reverse=True)
            beam = candidates[:k]
            
            # If all top k sequences ended with <eos>, stop
            if all(seq[-1] == vocab.word2id["<eos>"] for _, seq, _, _ in beam):
                break
                
        # Pick the best sequence
        best_seq = beam[0][1]
        
    words = []
    for i in best_seq[1:]:
        if i == vocab.word2id["<eos>"]: break
        if i in [vocab.word2id["<sos>"], vocab.word2id["<pad>"]]: continue
        
        # Clean up the output for the user
        word = vocab.id2word.get(i, "")
        if word in ["<2am>", "<2om>", "<unk>"]: continue
        words.append(word)
        
    return " ".join(words)

try:
    model, vocab = load_model()
except Exception as e:
    st.error(f"Model loading failed. Ensure vocab.pkl and lstm_focus.pt are in the repo. Error: {e}")

LANG_NAME = {"om": "Afaan Oromoo", "am": "አማርኛ", "en": "English"}

col1, col2 = st.columns(2)
with col1:
    src = st.selectbox("From", ["om", "am", "en"], format_func=lambda x: LANG_NAME[x])
with col2:
    tgt = st.selectbox("To", ["am", "om", "en"], format_func=lambda x: LANG_NAME[x], index=1)

text = st.text_area("Input text", placeholder="Enter text to translate...", height=150)

if st.button("Translate 🚀"):
    if not text.strip():
        st.warning("Please enter some text to translate.")
    elif src == tgt:
        st.info("Source and target languages are the same.")
    else:
        with st.spinner("Saiko is translating..."):
            try:
                result = translate(model, vocab, text, tgt_lang=tgt)
                st.success(result)
            except Exception as e:
                st.error(f"Translation error: {e}")
