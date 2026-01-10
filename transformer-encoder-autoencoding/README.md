# Transformer Encoder Autoencoding: Professional App

## Overview
This project demonstrates Masked Language Modeling (MLM) using:
- **Custom Transformer Encoder** (for learning)
- **Pre-trained BERT** (for real-world results)

## Features
- Web app with Streamlit
- Choose between your model and BERT
- Visualize attention weights
- Compare outputs side-by-side
- Modular, extensible code

## Quick Start
1. Install requirements:
	```
	pip install -r requirements.txt
	```
2. (Optional) Train your own model:
	```
	python transformer-encoder-autoencoding/train_mlm_wiki.py
	```
3. Run the app:
	```
	streamlit run transformer-encoder-autoencoding/app.py
	```

## Usage
- Enter a sentence with a `[MASK]` token.
- Select the model (Student or BERT).
- Click Predict to see the output and attention heatmap.

## Project Structure
- `encoder.py`, `attention.py`, `positional_encoding.py`: Custom transformer code
- `train_mlm_wiki.py`: Training script for your model
- `app.py`: Professional Streamlit app
- `bert_streamlit_app.py`: Minimal BERT demo
- `requirements.txt`: All dependencies

## Screenshots
Add screenshots of the app here after running.

---

**For best results, train your model on more data!**
