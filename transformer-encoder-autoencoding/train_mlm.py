import torch
import torch.nn as nn
import torch.optim as optim
from encoder import TransformerEncoder
from utils import build_vocab, encode_batch, decode_batch
import random

from synthetic_samples import MLM_PAIRS

# Toy sample data (original, 10 examples)
data = MLM_PAIRS

vocab = build_vocab([s for s, _ in data] + [t for _, t in data])
vocab_size = len(vocab)
max_len = max(len(s.split()) for s, _ in data)

def main():
    d_model = 64
    n_heads = 4
    num_layers = 2
    d_ff = 128
    model = TransformerEncoder(vocab_size, d_model, n_heads, num_layers, d_ff, max_len)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 100
    for epoch in range(epochs):
        total_loss = 0
        for src, tgt in data:
            src_ids = encode_batch([src], vocab, max_len)
            tgt_ids = encode_batch([tgt], vocab, max_len)
            src_tensor = torch.tensor(src_ids)
            tgt_tensor = torch.tensor(tgt_ids)
            out, _ = model(src_tensor)
            out = out.view(-1, vocab_size)
            tgt_tensor = tgt_tensor.view(-1)
            loss = criterion(out, tgt_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(data):.4f}")
    # Test
    for src, _ in data:
        src_ids = encode_batch([src], vocab, max_len)
        src_tensor = torch.tensor(src_ids)
        out, attn = model(src_tensor)
        pred_ids = out.argmax(-1)
        print(f"Input: {src}")
        print(f"Output: {decode_batch(pred_ids.tolist(), vocab)}\n")

if __name__ == "__main__":
    main()
