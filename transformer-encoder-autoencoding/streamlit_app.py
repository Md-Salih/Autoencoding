import streamlit as st

import torch
from encoder import TransformerEncoder
from utils import encode_batch, decode_batch
import os

# Try to load trained model and vocab
def load_student_model():
    from save_load_model import load_model as load_ckpt
    model_path = "../results/model_wiki.pth"
    if not os.path.exists(model_path):
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

@st.cache_resource
def load_bert_model():
    try:
        from transformers import BertTokenizer, BertForMaskedLM
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading BERT model: {e}")
        return None, None

st.title("Transformer Encoder Autoencoding (Masked Language Model)")
st.write("Enter a sentence with a [MASK] token. The model will reconstruct the masked word.")

# Model selector
model_choice = st.radio(
    "Select Model:",
    ("Student Model (Custom Transformer)", "BERT (Pretrained)"),
    horizontal=True
)

user_input = st.text_input("Input Sentence", "The chef adds [MASK] to the soup")

if st.button("Reconstruct"): 
    if model_choice == "Student Model (Custom Transformer)":
        model, vocab, max_len = load_student_model()
        if model is not None and vocab is not None:
            model.eval()
            src_ids = encode_batch([user_input], vocab, max_len)
            src_tensor = torch.tensor(src_ids)
            with torch.no_grad():
                out, attn = model(src_tensor)
                pred_ids = out.argmax(-1)
                output = decode_batch(pred_ids.tolist(), vocab)[0]
            st.success(f"**Student Model Output:** {output}")
            st.write("### Attention Weights (Head 0)")
            import matplotlib.pyplot as plt
            import seaborn as sns
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(attn[0][0].detach().numpy(), annot=True, cmap='viridis', ax=ax)
            st.pyplot(fig)
    
    else:  # BERT Model
        bert_model, tokenizer = load_bert_model()
        if bert_model is not None and tokenizer is not None:
            # Replace [MASK] with BERT's mask token if needed
            input_text = user_input.replace("[MASK]", tokenizer.mask_token)
            
            # Tokenize and predict
            inputs = tokenizer(input_text, return_tensors="pt")
            bert_model.eval()
            with torch.no_grad():
                outputs = bert_model(**inputs)
                predictions = outputs.logits
            
            # Find mask token position and get top prediction
            mask_token_index = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
            if len(mask_token_index) > 0:
                predicted_token_id = predictions[0, mask_token_index].argmax(axis=-1)
                predicted_token = tokenizer.decode(predicted_token_id[0])
                
                # Get top 5 predictions
                top_5 = torch.topk(predictions[0, mask_token_index[0]], 5)
                top_5_tokens = [tokenizer.decode([idx.item()]) for idx in top_5.indices]
                top_5_probs = torch.softmax(top_5.values, dim=0).tolist()
                
                output_text = user_input.replace("[MASK]", predicted_token)
                st.success(f"**BERT Model Output:** {output_text}")
                
                st.write("### Top 5 Predictions:")
                for i, (token, prob) in enumerate(zip(top_5_tokens, top_5_probs), 1):
                    st.write(f"{i}. **{token.strip()}** ({prob*100:.2f}%)")
            else:
                st.error("No [MASK] token found in the input sentence.")
