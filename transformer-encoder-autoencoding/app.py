import os

import streamlit as st
import torch

from utils import decode_batch, encode_batch

st.set_page_config(page_title="Transformer Encoder Autoencoding App", layout="centered")

st.title("Transformer Encoder Autoencoding: Professional Demo")
st.write("""
This app demonstrates two approaches for Masked Language Modeling (MLM):
- **Student Model**: Your own Transformer Encoder, trained from scratch (for learning).
- **Pre-trained Models**: Industry-standard MLM models from Hugging Face (for real-world results).

Enter a sentence with a [MASK] token and select a model to see the prediction.
""")

user_input = st.text_input("Input Sentence", "Transformers use [MASK] attention")
model_choice = st.radio(
    "Choose Mode",
    ["Student Model (Your Transformer)", "Pre-trained Model (Hugging Face)"],
)


def _normalize_mask(text: str, mask_token: str) -> str:
    # Let users always type [MASK]; convert to the chosen model's actual token.
    return text.replace("[MASK]", mask_token).replace("<mask>", mask_token)


@st.cache_resource
def _get_fill_mask_pipeline(model_id: str):
    from transformers import pipeline

    return pipeline("fill-mask", model=model_id)


def _find_student_checkpoint() -> str | None:
    base_dir = os.path.dirname(__file__)
    candidates = [
        os.path.join(base_dir, "results", "model_wiki.pth"),  # preferred
        os.path.abspath(os.path.join(base_dir, "..", "results", "model_wiki.pth")),  # legacy
        os.path.abspath(os.path.join(os.getcwd(), "results", "model_wiki.pth")),  # fallback
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

if model_choice.startswith("Pre-trained"):
    pretrained_options = {
        "BERT Base (bert-base-uncased)": "bert-base-uncased",
        "DistilBERT (distilbert-base-uncased)": "distilbert-base-uncased",
        "RoBERTa Base (roberta-base)": "roberta-base",
        "BERT Large (bert-large-uncased)": "bert-large-uncased",
    }
    selected_label = st.selectbox("Pre-trained Model", list(pretrained_options.keys()), index=0)
    top_k = st.slider("Top-K Predictions", min_value=1, max_value=10, value=5)

    st.markdown("#### Pre-download models (optional)")
    st.caption(
        "This will download the selected pre-trained models into your local Hugging Face cache so they load faster later. "
        "It may take several GB and a few minutes."
    )
    confirm_download = st.checkbox("I understand this will download large model files")
    if st.button("Download/Cache all pre-trained models", disabled=not confirm_download):
        model_ids = list(pretrained_options.values())
        with st.status("Downloading/caching models...", expanded=True) as status:
            for mid in model_ids:
                st.write(f"Caching: `{mid}`")
                nlp = _get_fill_mask_pipeline(mid)
                mask_token = nlp.tokenizer.mask_token
                sample = f"Transformers use {mask_token} attention"
                _ = nlp(sample, top_k=1)
            status.update(label="All selected models are cached.", state="complete")

if st.button("Predict"):
    if model_choice.startswith("Pre-trained"):
        model_id = pretrained_options[selected_label]
        nlp = _get_fill_mask_pipeline(model_id)
        mask_token = nlp.tokenizer.mask_token
        if not mask_token:
            st.error("Selected model does not support masked language modeling.")
        else:
            normalized = _normalize_mask(user_input, mask_token)
            if mask_token not in normalized:
                st.error("Please include a [MASK] token in the input sentence.")
            else:
                results = nlp(normalized, top_k=top_k)
                st.write(f"### {selected_label} Predictions")
                for res in results:
                    st.success(f"{res['sequence']} (score: {res['score']:.4f})")
    else:
        from encoder import TransformerEncoder

        ckpt_path = _find_student_checkpoint()
        if ckpt_path is None:
            st.error(
                "Student model checkpoint not found. Train it first using: "
                "`python transformer-encoder-autoencoding/train_mlm_wiki.py`"
            )
        else:
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            vocab = checkpoint["vocab"]
            vocab_size = len(vocab)

            d_model = 64
            n_heads = 4
            num_layers = 2
            d_ff = 128
            max_len = 20

            model = TransformerEncoder(vocab_size, d_model, n_heads, num_layers, d_ff, max_len)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            src_ids = encode_batch([user_input], vocab, max_len)
            src_tensor = torch.tensor(src_ids)
            with torch.no_grad():
                out, attn = model(src_tensor)
                pred_ids = out.argmax(-1)
                output = decode_batch(pred_ids.tolist(), vocab)[0]

            st.success(f"Student Model Output: {output}")
            st.caption(f"Checkpoint: {ckpt_path}")

            st.write("### Attention Weights (Head 0)")
            import matplotlib.pyplot as plt
            import seaborn as sns

            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(attn[0][0].detach().numpy(), annot=False, cmap="viridis", ax=ax)
            st.pyplot(fig)

st.markdown("---")
st.markdown("**Compare the results and explore how self-attention works!**")
