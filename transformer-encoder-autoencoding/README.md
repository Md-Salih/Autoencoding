# Transformer Encoder – Autoencoding (Masked Language Model)

## Objective
Understand Transformer Encoder, Self-Attention, and Autoencoding by:
- Reconstructing masked text (MLM)
- Using the same encoder for sentence classification

## What This Repo Covers (Checklist)
- Transformer Encoder (PyTorch): ✅ `encoder.py`, `attention.py`, `positional_encoding.py`
- Masked Language Modeling (MLM): ✅ `train_mlm.py` (toy dataset), ✅ `train_mlm_wiki.py` (Wikipedia)
- Attention visualization: ✅ `visualize_attention.ipynb` + generated PNG (below)
- Compare vs feed-forward baseline: ✅ `feedforward_baseline.py`
- Sentence classification using same encoder: ✅ `train_cls.py`

## Encoder Architecture Diagram
![Encoder Architecture](results/encoder_architecture.png)

## Autoencoding (MLM) Explanation
Autoencoding here means: you corrupt a sentence by replacing a word with `[MASK]`, then train the encoder to reconstruct the original sentence.
The Transformer Encoder uses self-attention to use *all* tokens as context for predicting the missing word.

## Sample Input/Output (Expected Reconstruction)
| Masked Input | Expected Output |
|---|---|
| Transformers use [MASK] attention | Transformers use self attention |
| Mars is called the [MASK] planet | Mars is called the red planet |
| Online learning improves [MASK] access | Online learning improves educational access |
| Exercise improves [MASK] health | Exercise improves mental health |
| Cricket is a [MASK] sport | Cricket is a popular sport |
| Python is a [MASK] language | Python is a programming language |
| Neural networks have [MASK] layers | Neural networks have hidden layers |
| Trees reduce [MASK] pollution | Trees reduce air pollution |
| Robots perform [MASK] tasks | Robots perform repetitive tasks |
| Solar power is a [MASK] source | Solar power is a renewable source |

## Attention Heatmap Screenshot
![Attention Heatmap](results/attention_heatmap.png)

## Sentence Classification (Same Encoder)
This project includes a tiny demo showing how the *same encoder* can do classification by adding a classifier head and using the first token representation ("[CLS]").

Run:
```
python transformer-encoder-autoencoding/train_cls.py
```

## Professional Demo App (Streamlit)
Run the unified app (Student model + BERT):
```
streamlit run transformer-encoder-autoencoding/app.py
```

## Setup
Install deps:
```
pip install -r transformer-encoder-autoencoding/requirements.txt
```

## Generate README Assets (PNG)
If you don’t see the images rendering, regenerate them:
```
python transformer-encoder-autoencoding/generate_assets.py
```
