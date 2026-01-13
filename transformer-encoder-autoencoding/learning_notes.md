# Learning Notes: Transformer Encoder Autoencoding Lab

## Key Concepts

### 1. Transformer Encoder
- Uses self-attention to process all tokens in parallel.
- Captures global context (every word can attend to every other word).
- No recurrence (unlike RNNs) and no convolution (unlike CNNs).

### 2. Self-Attention
- Computes attention scores between all pairs of words.
- Allows the model to focus on relevant words for each prediction.
- Multi-head: Multiple attention mechanisms run in parallel, capturing different relationships.

### 3. Autoencoding (Masked Language Modeling)
- Input: Sentence with some words replaced by [MASK].
- Model predicts the original words at masked positions.
- Trains the encoder to understand context and reconstruct missing information.

### 4. Feed-Forward Baseline
- Simple model: Embedding + fully connected layers.
- Cannot capture long-range dependencies as well as self-attention.

## Practical Steps
- **Build vocabulary** from all input/output sentences.
- **Encode** sentences as integer sequences.
- **Train** Transformer Encoder and Feed-Forward models to reconstruct masked sentences.
- **Visualize** attention weights to see which words the model focuses on.

## Insights
- Self-attention enables the model to use information from the entire sentence.
- Transformer Encoder outperforms feed-forward networks on masked word prediction.
- Attention heatmaps help interpret model decisions.

## Further Reading
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

---

**Tip:** Run `train_mlm.py` and `feedforward_baseline.py` to compare results. Use `visualize_attention.ipynb` to explore attention weights.
