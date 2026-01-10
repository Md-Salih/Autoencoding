import streamlit as st
import torch
from utils import encode_batch, decode_batch
import os

st.set_page_config(page_title="Transformer Encoder Autoencoding App", layout="centered")

st.title("Transformer Encoder Autoencoding: Professional Demo")
st.write("""
This app demonstrates two approaches for Masked Language Modeling (MLM):
- **Student Model**: Your own Transformer Encoder, trained from scratch (for learning).
- **BERT Model**: Industry-standard, pre-trained BERT from Hugging Face (for real-world results).

Enter a sentence with a [MASK] token and select a model to see the prediction.
""")

user_input = st.text_input("Input Sentence", "Transformers use [MASK] attention")
model_choice = st.radio("Choose Model", ["Student Model (Your Transformer)", "BERT Model (Pretrained)"])

if st.button("Predict"): 
    if model_choice.startswith("BERT"):
        from transformers import pipeline
        nlp = pipeline('fill-mask', model='bert-base-uncased')
        results = nlp(user_input)
        st.write("### BERT Top Predictions:")
        for res in results:
            st.success(f"{res['sequence']} (score: {res['score']:.4f})")
    else:
        # Student model
        from encoder import TransformerEncoder
        import pickle
        # Load model and vocab
        from save_load_model import load_model
        model_path = "results/model_wiki.pth"
        if not os.path.exists(model_path):
            st.error("Student model not trained yet. Please train it first.")
        else:
            checkpoint = torch.load(model_path, map_location='cpu')
            vocab = checkpoint['vocab']
            vocab_size = len(vocab)
            d_model = 64
            n_heads = 4
            num_layers = 2
            d_ff = 128
            max_len = 20
            model = TransformerEncoder(vocab_size, d_model, n_heads, num_layers, d_ff, max_len)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            src_ids = encode_batch([user_input], vocab, max_len)
            src_tensor = torch.tensor(src_ids)
            with torch.no_grad():
                out, attn = model(src_tensor)
                pred_ids = out.argmax(-1)
                output = decode_batch(pred_ids.tolist(), vocab)[0]
            st.success(f"Student Model Output: {output}")
            st.write("### Attention Weights (Head 0)")
            import matplotlib.pyplot as plt
            import seaborn as sns
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(attn[0][0].detach().numpy(), annot=False, cmap='viridis', ax=ax)
            st.pyplot(fig)

st.markdown("---")
st.markdown("**Compare the results and explore how self-attention works!**")
