import streamlit as st
import torch
from encoder import TransformerEncoder
from utils import encode_batch
import os

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Transformer Encoder Autoencoding",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= GLOBAL STYLES =================
st.markdown("""
<style>
body, .main { background-color: #0e1117; color: #ffffff; }

h1, h2, h3 { color: #ffffff; }

.app-card {
    background: #161a23;
    padding: 25px;
    border-radius: 14px;
    box-shadow: 0 0 30px rgba(0,0,0,0.55);
    margin-bottom: 20px;
}

.section-title {
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 10px;
}

input {
    background-color: #1f2430 !important;
    color: white !important;
}

.stButton > button {
    background: linear-gradient(135deg, #7f00ff, #a020f0);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
}

.stButton > button:hover {
    opacity: 0.85;
}

.result-box {
    background: #10131a;
    padding: 15px;
    border-radius: 10px;
    margin-top: 10px;
}

.footer {
    text-align: center;
    color: gray;
    margin-top: 50px;
    font-size: 13px;
}

/* Tooltip */
.info-icon {
    position: relative;
    display: inline-block;
    cursor: help;
    margin-left: 5px;
    color: #66aaff;
    font-size: 15px;
}
.info-icon .tooltiptext {
    visibility: hidden;
    background-color: #333;
    color: #fff;
    border-radius: 6px;
    padding: 8px 12px;
    position: absolute;
    z-index: 1000;
    bottom: 125%;
    right: 0;
    opacity: 0;
    transition: opacity 0.3s;
    white-space: nowrap;
}
.info-icon:hover .tooltiptext {
    visibility: visible;
    opacity: 1;
}
</style>
""", unsafe_allow_html=True)

# ================= MODEL LOADING (UNCHANGED) =================
def load_student_model():
    from save_load_model import load_model as load_ckpt

    model_paths = [
        "results/model_comprehensive.pth",
        "../results/model_comprehensive.pth",
        "results/model_wiki_robust.pth",
        "../results/model_wiki_robust.pth",
        "results/model_wiki_improved.pth",
        "../results/model_wiki_improved.pth",
        "../results/model_wiki.pth",
        "results/model_wiki.pth"
    ]

    checkpoint = None
    for path in model_paths:
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location="cpu")
            break

    if checkpoint is None:
        st.warning("Trained model not found.")
        return None, None, 20

    vocab = checkpoint["vocab"]
    config = checkpoint.get("config", {})

    model = TransformerEncoder(
        len(vocab),
        config.get("d_model", 64),
        config.get("n_heads", 4),
        config.get("n_layers", 2),
        config.get("d_ff", 128),
        checkpoint.get("max_len", 20)
    )

    model.load_state_dict(
        checkpoint.get("model_state_dict", checkpoint.get("model_state"))
    )

    return model, vocab, checkpoint.get("max_len", 20)

# ================= APP HEADER =================
st.markdown("""
<div class="app-card">
<h1>Transformer Encoder Autoencoding</h1>
<p>Masked Language Modeling with Student & Pretrained Models</p>
</div>
""", unsafe_allow_html=True)

# ================= MAIN LAYOUT =================
left, right = st.columns([1, 2])

# ================= LEFT PANEL =================
with left:
    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>⚙️ Model Selection</div>", unsafe_allow_html=True)

    model_choice = st.selectbox(
        "Choose Model",
        [
            "Student Model (Custom Trained)",
            "BERT Base (110M params)",
            "RoBERTa Base (125M params)",
            "DistilBERT (66M params - Fast)"
        ]
    )

    temperature = 1.0
    top_k_display = 5

    if model_choice != "Student Model (Custom Trained)":
        temperature = st.slider("Temperature", 0.1, 2.0, 1.0, 0.1)
        top_k_display = st.slider("Top-K Predictions", 1, 10, 5)

    st.markdown("</div>", unsafe_allow_html=True)

# ================= RIGHT PANEL =================
with right:
    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Input Sentence</div>", unsafe_allow_html=True)

    user_input = st.text_input(
        "Sentence",
        placeholder="Example: The chef adds [MASK] to the soup"
    )

    run = st.button("🔍 Reconstruct")

    st.markdown("</div>", unsafe_allow_html=True)
    
    # ================= RESULTS (Inside Right Panel) =================
    if run and user_input.strip():
        if "[MASK]" not in user_input:
            st.error("Input must contain a [MASK] token.")
        elif model_choice == "Student Model (Custom Trained)":
            with st.spinner("Loading student model..."):
                model, vocab, max_len = load_student_model()

            if model:
                model.eval()
                words = user_input.split()
                
                if "[MASK]" not in words:
                    st.error("[MASK] token not found in input.")
                else:
                    mask_idx = words.index("[MASK]")

                    src_ids = encode_batch([user_input], vocab, max_len)
                    src_tensor = torch.tensor(src_ids)

                    with torch.no_grad():
                        out, _ = model(src_tensor)
                        probs = torch.softmax(out[0, mask_idx], dim=0)
                        top_probs, top_indices = probs.topk(top_k_display)

                    idx_to_word = {v: k for k, v in vocab.items()}
                    predictions = [idx_to_word[i.item()] for i in top_indices]

                    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
                    st.markdown("<div class='section-title'>Prediction Results</div>", unsafe_allow_html=True)

                    for i, (word, prob) in enumerate(zip(predictions, top_probs), 1):
                        sentence = user_input.replace("[MASK]", f"**{word}**")
                        st.write(f"{i}. {sentence}")
                        st.progress(prob.item(), text=f"Confidence: {prob.item()*100:.2f}%")

                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.error("Failed to load student model. Please train a model first.")
        
        else:
            # Pretrained models
            with st.spinner(f"🔄 Loading {model_choice}..."):
                try:
                    from transformers import AutoTokenizer, AutoModelForMaskedLM
                    import torch.nn.functional as F
                    
                    # Map model names to HuggingFace IDs
                    model_map = {
                        "BERT Base (110M params)": "bert-base-uncased",
                        "RoBERTa Base (125M params)": "roberta-base",
                        "DistilBERT (66M params - Fast)": "distilbert-base-uncased"
                    }
                    
                    model_name = model_map[model_choice]
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForMaskedLM.from_pretrained(model_name)
                    model.eval()
                    
                    # Convert [MASK] to the model's mask token format
                    model_input = user_input.replace("[MASK]", tokenizer.mask_token)
                    
                    # Tokenize input
                    inputs = tokenizer(model_input, return_tensors="pt")
                    
                    # Find mask token position
                    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
                    
                    if len(mask_token_index) == 0:
                        st.error(f"❌ Mask token not properly recognized. This model uses `{tokenizer.mask_token}`")
                    else:
                        # Get predictions
                        with torch.no_grad():
                            outputs = model(**inputs)
                            logits = outputs.logits
                            
                        # Get predictions for masked token
                        mask_token_logits = logits[0, mask_token_index, :]
                        
                        # Apply temperature
                        mask_token_logits = mask_token_logits / temperature
                        
                        # Get probabilities
                        probs = F.softmax(mask_token_logits, dim=-1)
                        
                        # Get top-k predictions
                        top_probs, top_indices = torch.topk(probs, top_k_display, dim=-1)
                        
                        # Decode predictions
                        st.markdown("<div class='app-card'>", unsafe_allow_html=True)
                        st.markdown("<div class='section-title'>✅ Prediction Results</div>", unsafe_allow_html=True)
                        
                        for i in range(top_k_display):
                            token_id = top_indices[0, i].item()
                            token = tokenizer.decode([token_id]).strip()
                            probability = top_probs[0, i].item()
                            
                            # Reconstruct sentence
                            sentence = user_input.replace("[MASK]", f"**{token}**")
                            st.write(f"{i+1}. {sentence}")
                            st.progress(probability, text=f"Confidence: {probability*100:.2f}%")
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ Error loading pretrained model: {str(e)}")
                    st.info("💡 Try installing transformers: `pip install transformers`")
                    st.info("Try installing transformers: `pip install transformers`")

    elif run:
        st.warning("Please enter a sentence with [MASK] token.")

# ================= FOOTER =================
st.markdown("<div class='footer'>Made with Streamlit • Transformer Encoder Demo</div>", unsafe_allow_html=True)
