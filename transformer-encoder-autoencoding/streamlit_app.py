import streamlit as st

import torch
from encoder import TransformerEncoder
from utils import encode_batch, decode_batch
import os

# Try to load trained model and vocab
def load_student_model():
    from save_load_model import load_model as load_ckpt
    
    # Try robust model first, then improved, then fall back to old model
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
    model_path_used = None
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            model_path_used = model_path
            break
    
    if checkpoint is None:
        st.warning("Trained model not found. Please run train_mlm_comprehensive.py first for best results.")
        return None, None, 20
    
    vocab = checkpoint['vocab']
    vocab_size = len(vocab)
    config = checkpoint.get('config', {})
    
    # Get model architecture from config (handle both 'n_layers' and 'num_layers')
    d_model = config.get('d_model', 64)
    n_heads = config.get('n_heads', 4)
    num_layers = config.get('n_layers', config.get('num_layers', 2))  # Check both keys
    d_ff = config.get('d_ff', 128)
    max_len = checkpoint.get('max_len', config.get('max_len', 20))
    
    # Show which model is being used with detailed info
    if "comprehensive" in model_path_used:
        model_type = "Comprehensive Dataset (Best)"
        st.success(f"✨ Using {model_type} Student Model")
        st.info(f"📊 Trained on 200+ diverse examples covering: general knowledge, daily life, technology, nature, education, health, business, travel, and sports!")
        st.caption(f"Architecture: d_model={d_model}, layers={num_layers}, vocab={vocab_size}")
    elif "robust" in model_path_used:
        model_type = "Robust (Good)"
        st.info(f"Using {model_type} Student Model (d_model={d_model}, layers={num_layers}, vocab={vocab_size})")
    elif "improved" in model_path_used:
        model_type = "Improved"
        st.info(f"Using {model_type} Student Model (d_model={d_model}, layers={num_layers}, vocab={vocab_size})")
    else:
        model_type = "Original"
        st.info(f"Using {model_type} Student Model (d_model={d_model}, layers={num_layers}, vocab={vocab_size})")
    
    model = TransformerEncoder(vocab_size, d_model, n_heads, num_layers, d_ff, max_len)
    
    # Handle both 'model_state_dict' and 'model_state' keys for compatibility
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        st.error("Model checkpoint format not recognized")
        return None, None, None
    
    return model, vocab, max_len

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

@st.cache_resource
def load_roberta_model():
    try:
        from transformers import RobertaTokenizer, RobertaForMaskedLM
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForMaskedLM.from_pretrained('roberta-base')
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading RoBERTa model: {e}")
        return None, None

@st.cache_resource
def load_distilbert_model():
    try:
        from transformers import DistilBertTokenizer, DistilBertForMaskedLM
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading DistilBERT model: {e}")
        return None, None

def predict_with_pretrained(model, tokenizer, input_text, temperature=1.0, top_k=5):
    """Predict with adjustable temperature for diversity"""
    # Replace [MASK] with model's mask token
    input_text = input_text.replace("[MASK]", tokenizer.mask_token)
    
    inputs = tokenizer(input_text, return_tensors="pt")
    model.eval()
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits
    
    # Find mask token position
    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
    
    if len(mask_token_index) == 0:
        return None, None, None
    
    # Apply temperature scaling
    logits = predictions[0, mask_token_index[0]] / temperature
    probs = torch.softmax(logits, dim=0)
    
    # Get top k predictions and sort by probability (descending)
    top_k_results = torch.topk(probs, min(top_k, len(probs)))
    top_k_tokens = [tokenizer.decode([idx.item()]) for idx in top_k_results.indices]
    top_k_probs = top_k_results.values.tolist()
    
    # Ensure sorted by probability (highest first)
    sorted_pairs = sorted(zip(top_k_tokens, top_k_probs), key=lambda x: x[1], reverse=True)
    top_k_tokens = [t for t, _ in sorted_pairs]
    top_k_probs = [p for _, p in sorted_pairs]
    
    # Get top prediction
    predicted_token = top_k_tokens[0]
    output_text = input_text.replace(tokenizer.mask_token, predicted_token)
    
    return output_text, top_k_tokens, top_k_probs

def get_training_examples():
    """Get sample training examples from the model checkpoint"""
    model_paths = [
        "results/model_wiki_improved.pth",
        "../results/model_wiki_improved.pth",
        "../results/model_wiki.pth",
        "results/model_wiki.pth"
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            config = checkpoint.get('config', {})
            
            # Sample sentences that work well with the model
            examples = [
                "The chef adds [MASK] to the soup",
                "I love to eat [MASK] food",
                "The cat is [MASK] on the couch",
                "She reads a [MASK] every day",
                "The sun is [MASK] in the sky",
                "He plays [MASK] with friends",
                "The dog runs in the [MASK]",
                "She drinks [MASK] in the morning",
                "The [MASK] is very beautiful",
                "They are [MASK] to school",
            ]
            
            return examples, config
    return None, None

st.title("Transformer Encoder Autoencoding (Masked Language Model)")
st.write("Enter a sentence with a [MASK] token. The model will reconstruct the masked word.")

# Model selector
model_choice = st.selectbox(
    "Select Model:",
    [
        "Student Model (Custom Trained)",
        "BERT Base (110M params)",
        "RoBERTa Base (125M params)", 
        "DistilBERT (66M params - Fast)"
    ]
)

# Temperature slider for pretrained models only
temperature = 1.0
top_k_display = 5

if model_choice != "Student Model (Custom Trained)":
    st.sidebar.header("🎛️ Prediction Settings")
    st.sidebar.write("Adjust how creative or conservative the predictions are:")
    
    temperature = st.sidebar.slider(
        "Temperature (Diversity)",
        min_value=0.1,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="Lower = more confident/conservative, Higher = more diverse/creative"
    )
    
    top_k_display = st.sidebar.slider(
        "Number of Predictions",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        help="How many alternative predictions to show"
    )
    
    st.sidebar.info(
        "💡 **Tip:**\n"
        "- Temperature 0.5-0.7: Safe, accurate predictions\n"
        "- Temperature 1.0: Balanced (default)\n"
        "- Temperature 1.5-2.0: Creative, diverse predictions"
    )

user_input = st.text_input("Input Sentence", placeholder="Enter a sentence with [MASK] token")

if st.button("Reconstruct"): 
    if model_choice == "Student Model (Custom Trained)":
        model, vocab, max_len = load_student_model()
        if model is not None and vocab is not None:
            # Check if [MASK] exists in input
            if "[MASK]" not in user_input:
                st.error("No [MASK] token found in the input sentence.")
            else:
                model.eval()
                
                # Find the position of [MASK]
                words = user_input.split()
                try:
                    mask_idx = words.index("[MASK]")
                except ValueError:
                    st.error("Could not find [MASK] token")
                    st.stop()
                
                # Encode input
                src_ids = encode_batch([user_input], vocab, max_len)
                src_tensor = torch.tensor(src_ids)
                
                with torch.no_grad():
                    out, attn = model(src_tensor)  # [B, T, vocab_size]
                    
                    # Get predictions at mask position
                    mask_logits = out[0, mask_idx]  # [vocab_size]
                    
                    # Apply temperature
                    scaled_logits = mask_logits / temperature
                    probs = torch.softmax(scaled_logits, dim=0)
                    
                    # Get top-k predictions
                    top_probs, top_indices = probs.topk(top_k_display)
                    
                    # Convert to words
                    idx_to_word = {idx: word for word, idx in vocab.items()}
                    top_words = [idx_to_word.get(idx.item(), "<UNK>") for idx in top_indices]
                    
                    # Create output sentence with top prediction
                    output_words = words.copy()
                    output_words[mask_idx] = top_words[0]
                    output = " ".join(output_words)
                
                st.success(f"**Student Model Output:** {output}")
                
                # Show top predictions with same style as pretrained models
                st.markdown("""
                <style>
                .info-icon {
                    position: relative;
                    display: inline-block;
                    cursor: help;
                    margin-left: 5px;
                    color: #0066cc;
                    font-size: 16px;
                }
                .info-icon .tooltiptext {
                    visibility: hidden;
                    background-color: #555;
                    color: #fff;
                    text-align: center;
                    border-radius: 6px;
                    padding: 10px 15px;
                    position: absolute;
                    z-index: 1000;
                    bottom: 125%;
                    right: 0;
                    transform: translateX(0);
                    opacity: 0;
                    transition: opacity 0.3s;
                    white-space: nowrap;
                    font-size: 14px;
                    margin-bottom: 5px;
                }
                .info-icon:hover .tooltiptext {
                    visibility: visible;
                    opacity: 1;
                }
                </style>
                """, unsafe_allow_html=True)
                
                st.write(f"### Top {len(top_words)} Predictions")
                for i, (word, prob) in enumerate(zip(top_words, top_probs), 1):
                    # Create the full sentence with this prediction
                    full_sentence = user_input.replace("[MASK]", word.strip())
                    
                    col1, col2, col3 = st.columns([0.1, 0.7, 0.2])
                    with col1:
                        st.write(f"**{i}.**")
                    with col2:
                        st.progress(prob.item(), text=f"**{word.strip()}**")
                    with col3:
                        st.markdown(f'**{prob.item()*100:.2f}%** <span class="info-icon">ℹ️<span class="tooltiptext">{full_sentence}</span></span>', unsafe_allow_html=True)
    
    elif model_choice == "BERT Base (110M params)":
        bert_model, tokenizer = load_bert_model()
        if bert_model is not None and tokenizer is not None:
            output_text, top_tokens, top_probs = predict_with_pretrained(
                bert_model, tokenizer, user_input, temperature, top_k_display
            )
            
            if output_text:
                st.success(f"**BERT Output:** {output_text}")
                
                # Add CSS for icon-based tooltip
                st.markdown("""
                <style>
                .info-icon {
                    position: relative;
                    display: inline-block;
                    cursor: help;
                    margin-left: 5px;
                    color: #0066cc;
                    font-size: 16px;
                }
                .info-icon .tooltiptext {
                    visibility: hidden;
                    background-color: #555;
                    color: #fff;
                    text-align: center;
                    border-radius: 6px;
                    padding: 10px 15px;
                    position: absolute;
                    z-index: 1000;
                    bottom: 125%;
                    right: 0;
                    transform: translateX(0);
                    opacity: 0;
                    transition: opacity 0.3s;
                    white-space: nowrap;
                    font-size: 14px;
                    margin-bottom: 5px;
                }
                .info-icon:hover .tooltiptext {
                    visibility: visible;
                    opacity: 1;
                }
                </style>
                """, unsafe_allow_html=True)
                
                st.write(f"### Top {len(top_tokens)} Predictions")
                for i, (token, prob) in enumerate(zip(top_tokens, top_probs), 1):
                    # Create the full sentence with this prediction
                    full_sentence = user_input.replace("[MASK]", token.strip())
                    
                    col1, col2, col3 = st.columns([0.1, 0.7, 0.2])
                    with col1:
                        st.write(f"**{i}.**")
                    with col2:
                        st.progress(prob, text=f"**{token.strip()}**")
                    with col3:
                        st.markdown(f'**{prob*100:.2f}%** <span class="info-icon">ℹ️<span class="tooltiptext">{full_sentence}</span></span>', unsafe_allow_html=True)
            else:
                st.error("No [MASK] token found in the input sentence.")
    
    elif model_choice == "RoBERTa Base (125M params)":
        roberta_model, tokenizer = load_roberta_model()
        if roberta_model is not None and tokenizer is not None:
            output_text, top_tokens, top_probs = predict_with_pretrained(
                roberta_model, tokenizer, user_input, temperature, top_k_display
            )
            
            if output_text:
                st.success(f"**RoBERTa Output:** {output_text}")
                
                # Add CSS for icon-based tooltip
                st.markdown("""
                <style>
                .info-icon {
                    position: relative;
                    display: inline-block;
                    cursor: help;
                    margin-left: 5px;
                    color: #0066cc;
                    font-size: 16px;
                }
                .info-icon .tooltiptext {
                    visibility: hidden;
                    background-color: #555;
                    color: #fff;
                    text-align: center;
                    border-radius: 6px;
                    padding: 10px 15px;
                    position: absolute;
                    z-index: 1000;
                    bottom: 125%;
                    right: 0;
                    transform: translateX(0);
                    opacity: 0;
                    transition: opacity 0.3s;
                    white-space: nowrap;
                    font-size: 14px;
                    margin-bottom: 5px;
                }
                .info-icon:hover .tooltiptext {
                    visibility: visible;
                    opacity: 1;
                }
                </style>
                """, unsafe_allow_html=True)
                
                st.write(f"### Top {len(top_tokens)} Predictions")
                for i, (token, prob) in enumerate(zip(top_tokens, top_probs), 1):
                    # Create the full sentence with this prediction
                    full_sentence = user_input.replace("[MASK]", token.strip())
                    
                    col1, col2, col3 = st.columns([0.1, 0.7, 0.2])
                    with col1:
                        st.write(f"**{i}.**")
                    with col2:
                        st.progress(prob, text=f"**{token.strip()}**")
                    with col3:
                        st.markdown(f'**{prob*100:.2f}%** <span class="info-icon">ℹ️<span class="tooltiptext">{full_sentence}</span></span>', unsafe_allow_html=True)
            else:
                st.error("No [MASK] token found in the input sentence.")
    
    elif model_choice == "DistilBERT (66M params - Fast)":
        distilbert_model, tokenizer = load_distilbert_model()
        if distilbert_model is not None and tokenizer is not None:
            output_text, top_tokens, top_probs = predict_with_pretrained(
                distilbert_model, tokenizer, user_input, temperature, top_k_display
            )
            
            if output_text:
                st.success(f"**DistilBERT Output:** {output_text}")
                
                # Add CSS for icon-based tooltip
                st.markdown("""
                <style>
                .info-icon {
                    position: relative;
                    display: inline-block;
                    cursor: help;
                    margin-left: 5px;
                    color: #0066cc;
                    font-size: 16px;
                }
                .info-icon .tooltiptext {
                    visibility: hidden;
                    background-color: #555;
                    color: #fff;
                    text-align: center;
                    border-radius: 6px;
                    padding: 10px 15px;
                    position: absolute;
                    z-index: 1000;
                    bottom: 125%;
                    right: 0;
                    transform: translateX(0);
                    opacity: 0;
                    transition: opacity 0.3s;
                    white-space: nowrap;
                    font-size: 14px;
                    margin-bottom: 5px;
                }
                .info-icon:hover .tooltiptext {
                    visibility: visible;
                    opacity: 1;
                }
                </style>
                """, unsafe_allow_html=True)
                
                st.write(f"### Top {len(top_tokens)} Predictions")
                for i, (token, prob) in enumerate(zip(top_tokens, top_probs), 1):
                    # Create the full sentence with this prediction
                    full_sentence = user_input.replace("[MASK]", token.strip())
                    
                    col1, col2, col3 = st.columns([0.1, 0.7, 0.2])
                    with col1:
                        st.write(f"**{i}.**")
                    with col2:
                        st.progress(prob, text=f"**{token.strip()}**")
                    with col3:
                        st.markdown(f'**{prob*100:.2f}%** <span class="info-icon">ℹ️<span class="tooltiptext">{full_sentence}</span></span>', unsafe_allow_html=True)
            else:
                st.error("No [MASK] token found in the input sentence.")
