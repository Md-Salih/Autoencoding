# Transformer Encoder Autoencoding: Professional App

Welcome! This app lets you explore Masked Language Modeling (MLM) using:
- **Your own Transformer Encoder** (for learning and experimentation)
- **Pre-trained BERT** (for industry-standard results)

## How to Use
1. Open the app with:
   ```
   streamlit run app.py
   ```
2. Enter a sentence with a `[MASK]` token (e.g., `The chef adds [MASK] to the soup`).
3. Choose which model to use.
4. Click Predict and see the results, including attention heatmaps for your model.

## Features
- Compare your model and BERT side-by-side
- Visualize attention weights
- Download predictions (coming soon)
- Professional UI/UX

## For Developers
- All code is modular and well-documented
- Easily extend with new models or features

---

**Tip:** For best results with your own model, train on more data using `train_mlm_wiki.py`.
