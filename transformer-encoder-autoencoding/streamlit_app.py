import streamlit as st

import torch
from encoder import TransformerEncoder
from utils import encode_batch, decode_batch
import os

# Try to load trained model and vocab
def load_model_and_vocab():
    from save_load_model import load_model as load_ckpt
    model_path = "results/model_wiki.pth"
    d_model = 64
    n_heads = 4
    num_layers = 2
    d_ff = 128
    max_len = 20  # fallback
    if os.path.exists(model_path):
        # Load vocab first to get vocab_size and max_len
        checkpoint = torch.load(model_path, map_location='cpu')
        vocab = checkpoint['vocab']
        vocab_size = len(vocab)
        # Estimate max_len from vocab (not perfect, but works for most cases)
        model = TransformerEncoder(vocab_size, d_model, n_heads, num_layers, d_ff, max_len)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model, vocab, max_len
    else:
        st.warning("Trained model not found. Please run train_mlm_wiki.py first.")
        return None, None, max_len

st.title("Transformer Encoder Autoencoding (Masked Language Model)")
st.write("Enter a sentence with a [MASK] token. The model will reconstruct the masked word.")

user_input = st.text_input("Input Sentence", "The chef adds [MASK] to the soup")

if st.button("Reconstruct"): 
    model, vocab, max_len = load_model_and_vocab()
    if model is not None and vocab is not None:
        model.eval()
        src_ids = encode_batch([user_input], vocab, max_len)
        src_tensor = torch.tensor(src_ids)
        with torch.no_grad():
            out, attn = model(src_tensor)
            pred_ids = out.argmax(-1)
            output = decode_batch(pred_ids.tolist(), vocab)[0]
        st.success(f"Reconstructed Output: {output}")
        st.write("### Attention Weights (Head 0)")
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(attn[0][0].detach().numpy(), annot=True, cmap='viridis', ax=ax)
        st.pyplot(fig)
