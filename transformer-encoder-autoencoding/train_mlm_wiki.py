import torch
import torch.nn as nn
import torch.optim as optim
import os
# HF token removed for security. No token is required for public datasets.
from encoder import TransformerEncoder
from utils import build_vocab, encode_batch, decode_batch
from data_loader import get_wiki_sentences
import random

def mask_sentence(sentence):
    words = sentence.split()
    if len(words) < 3:
        return None, None
    idx = random.randint(1, len(words)-2)
    masked = words.copy()
    masked[idx] = '[MASK]'
    return ' '.join(masked), sentence

def main():
    print("Loading Wikipedia sentences...")
    sentences = get_wiki_sentences(500)
    data = []
    for s in sentences:
        masked, orig = mask_sentence(s)
        if masked and orig:
            data.append((masked, orig))
    print(f"Prepared {len(data)} masked sentences.")
    vocab = build_vocab([s for s, _ in data] + [t for _, t in data])
    vocab_size = len(vocab)
    max_len = max(len(s.split()) for s, _ in data)
    d_model = 64
    n_heads = 4
    num_layers = 2
    d_ff = 128
    config = {
        "vocab_size": vocab_size,
        "d_model": d_model,
        "n_heads": n_heads,
        "num_layers": num_layers,
        "d_ff": d_ff,
        "max_len": max_len,
        "task": "mlm",
    }
    model = TransformerEncoder(vocab_size, d_model, n_heads, num_layers, d_ff, max_len)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 10
    for epoch in range(epochs):
        total_loss = 0
        for src, tgt in random.sample(data, min(100, len(data))):
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
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(data):.4f}")
    # Save model and vocab
    from save_load_model import save_model
    save_model(model, vocab, path="results/model_wiki.pth", config=config)
    # Test
    for src, _ in random.sample(data, 5):
        src_ids = encode_batch([src], vocab, max_len)
        src_tensor = torch.tensor(src_ids)
        out, attn = model(src_tensor)
        pred_ids = out.argmax(-1)
        print(f"Input: {src}")
        print(f"Output: {decode_batch(pred_ids.tolist(), vocab)}\n")

if __name__ == "__main__":
    main()
