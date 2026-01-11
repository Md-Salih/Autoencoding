import os
import sys
import subprocess
import time

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

user_input = st.text_input("Input Sentence", "The chef adds [MASK] to the soup")
model_choice = st.radio(
    "Choose Mode",
    ["Student Model (Your Transformer)", "Pre-trained Model (Hugging Face)"],
)


def _results_dir() -> str:
    base_dir = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(base_dir, "results")


def _tail_file(path: str, max_lines: int = 200) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return "".join(lines[-max_lines:])


def _start_training(script_name: str, log_name: str) -> None:
    os.makedirs(_results_dir(), exist_ok=True)
    base_dir = os.path.abspath(os.path.dirname(__file__))
    script_path = os.path.join(base_dir, script_name)
    log_path = os.path.join(_results_dir(), log_name)

    # Start a separate process so Streamlit doesn't freeze.
    # Redirect stdout/stderr to a log file that we can tail in the UI.
    log_file = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, script_path],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=base_dir,
    )
    st.session_state["train_proc"] = proc
    st.session_state["train_log"] = log_path
    st.session_state["train_started_at"] = time.time()


def _train_status():
    proc = st.session_state.get("train_proc")
    if proc is None:
        return None
    return proc.poll()  # None=running, else exit code


def _normalize_mask(text: str, mask_token: str) -> str:
    # Let users always type [MASK]; convert to the chosen model's actual token.
    return text.replace("[MASK]", mask_token).replace("<mask>", mask_token)


@st.cache_resource
def _get_fill_mask_pipeline(model_id: str):
    # Keep the app output clean: Transformers can emit expected warnings
    # about unused weights (pooler) when initializing MLM heads.
    from transformers import logging as hf_logging, pipeline

    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()

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


def _find_student_checkpoints() -> dict:
    base_dir = os.path.dirname(__file__)
    candidates = {
        "Student (Lab-trained)": os.path.join(base_dir, "results", "model_lab.pth"),
        "Student (Wiki-trained)": os.path.join(base_dir, "results", "model_wiki.pth"),
    }
    found = {}
    for label, path in candidates.items():
        if os.path.exists(path):
            found[label] = path
    return found

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

if model_choice.startswith("Student"):
    found = _find_student_checkpoints()
    student_options = [
        "Student (Lab-trained)",
        "Student (Wiki-trained)",
    ]

    available = [opt for opt in student_options if opt in found]
    if len(available) == 1:
        student_selected = available[0]
        st.text_input("Student Checkpoint", value=student_selected, disabled=True)
    else:
        student_selected = st.selectbox(
            "Student Checkpoint",
            student_options,
            index=0,
        )
    if student_selected not in found:
        st.warning(
            "Selected checkpoint not found yet. Train it first.\n\n"
            "- Lab: `python transformer-encoder-autoencoding/train_mlm_lab.py`\n"
            "- Wiki: `python transformer-encoder-autoencoding/train_mlm_wiki.py`"
        )

    with st.expander("Train Student Model (from the app)", expanded=False):
        st.caption(
            "Starts training in a separate background process and writes logs to `transformer-encoder-autoencoding/results/`. "
            "Lab training is fast; Wikipedia training can take a long time and download data."
        )

        running = _train_status() is None and st.session_state.get("train_proc") is not None
        cols = st.columns(3)

        with cols[0]:
            if st.button("Train Lab (Synthetic 10)", disabled=running):
                _start_training("train_mlm_lab.py", "train_lab.log")
                st.success("Started Lab training.")

        with cols[1]:
            if st.button("Train Wiki (Wikipedia)", disabled=running):
                _start_training("train_mlm_wiki.py", "train_wiki.log")
                st.success("Started Wiki training.")

        with cols[2]:
            if st.button("Stop Training", disabled=not running):
                proc = st.session_state.get("train_proc")
                if proc is not None:
                    proc.terminate()
                st.warning("Stop requested.")

        log_path = st.session_state.get("train_log")
        if log_path:
            exit_code = _train_status()
            if exit_code is None:
                st.info("Training is running...")
            else:
                st.success(f"Training finished (exit code: {exit_code}).")
            st.code(_tail_file(log_path), language="text")

    st.markdown("#### Student prediction mode")
    student_fill_mode = st.radio(
        "How should the Student model produce output?",
        ["Fill [MASK] only (recommended)", "Reconstruct all tokens"],
        index=0,
        help="Fill-only keeps your original words and replaces just the [MASK] token(s), which is much more stable for unseen inputs.",
    )
    student_top_k = st.slider("Student Top-K (per mask)", min_value=1, max_value=10, value=5)

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

        found = _find_student_checkpoints()
        ckpt_path = found.get(student_selected)
        if ckpt_path is None:
            st.error("Selected student checkpoint is missing. Train it first.")
        else:
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            vocab = checkpoint["vocab"]
            vocab_size = len(vocab)
            cfg = checkpoint.get("config", {})

            d_model = int(cfg.get("d_model", 64))
            n_heads = int(cfg.get("n_heads", 4))
            num_layers = int(cfg.get("num_layers", 2))
            d_ff = int(cfg.get("d_ff", 128))
            max_len = int(cfg.get("max_len", 20))

            model = TransformerEncoder(vocab_size, d_model, n_heads, num_layers, d_ff, max_len)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            # Help users understand why Wiki-student can look "wrong" on arbitrary inputs.
            raw_tokens = user_input.split()
            unknown_tokens = [t for t in raw_tokens if t not in vocab and t != "[MASK]"]
            if unknown_tokens:
                st.warning(
                    "Student vocab is limited (whitespace tokenizer). Some input words are out-of-vocabulary, "
                    "so the Wiki student may behave poorly on full-sentence reconstruction. "
                    "Use 'Fill [MASK] only' or switch to Pre-trained mode for best results.\n\n"
                    f"Unknown words ({len(unknown_tokens)}): {', '.join(unknown_tokens[:12])}"
                )

            tokens = user_input.split()
            if len(tokens) > max_len:
                st.warning(f"Input has {len(tokens)} tokens but this checkpoint supports max_len={max_len}. Extra tokens will be truncated.")

            src_ids = encode_batch([user_input], vocab, max_len)
            src_tensor = torch.tensor(src_ids)

            with torch.no_grad():
                out, attn = model(src_tensor)

            inv_vocab = {v: k for k, v in vocab.items()}

            if "[MASK]" not in tokens:
                st.error("Please include a [MASK] token in the input sentence.")
            else:
                if student_fill_mode.startswith("Fill"):
                    output_tokens = tokens[:]
                    mask_positions = [i for i, t in enumerate(tokens) if t == "[MASK]" and i < max_len]

                    all_suggestions = []
                    for pos in mask_positions:
                        logits = out[0, pos]  # [vocab]
                        topk = torch.topk(logits, k=int(student_top_k))
                        ids = topk.indices.tolist()
                        vals = topk.values.tolist()
                        suggestions = [(inv_vocab.get(i, "[UNK]"), float(v)) for i, v in zip(ids, vals)]
                        all_suggestions.append((pos, suggestions))

                        # Best token
                        best_token = suggestions[0][0] if suggestions else "[UNK]"
                        output_tokens[pos] = best_token

                    output = " ".join(output_tokens)
                    st.success(f"Student Model Output: {output}")

                    st.write("### Student Top-K suggestions")
                    for pos, suggestions in all_suggestions:
                        st.caption(f"Mask position {pos}")
                        st.write(", ".join([f"{tok}" for tok, _ in suggestions]))
                else:
                    pred_ids = out.argmax(-1)
                    output = decode_batch(pred_ids.tolist(), vocab)[0]
                    st.success(f"Student Model Output (reconstructed): {output}")

            st.caption(f"Checkpoint: {ckpt_path}")

            st.write("### Attention Weights (Head 0)")
            import matplotlib.pyplot as plt
            import seaborn as sns

            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(attn[0][0].detach().numpy(), annot=False, cmap="viridis", ax=ax)
            st.pyplot(fig)

st.markdown("---")
st.markdown("**Compare the results and explore how self-attention works!**")
