import torch
import torch.nn as nn
import torch.optim as optim
from utils import build_vocab, encode_batch, decode_batch

# Simple feed-forward baseline for MLM
class FeedForwardMLM(nn.Module):
    def __init__(self, vocab_size, d_model, max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.fc = nn.Sequential(
            nn.Linear(d_model * max_len, 128),
            nn.ReLU(),
            nn.Linear(128, vocab_size * max_len)
        )
        self.max_len = max_len
        self.vocab_size = vocab_size

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        out = out.view(-1, self.max_len, self.vocab_size)
        return out

def main():
    data = [
        ("Transformers use [MASK] attention", "Transformers use self attention"),
        ("Mars is called the [MASK] planet", "Mars is called the red planet"),
        ("Online learning improves [MASK] access", "Online learning improves educational access"),
        ("Exercise improves [MASK] health", "Exercise improves mental health"),
        ("Cricket is a [MASK] sport", "Cricket is a popular sport"),
        ("Python is a [MASK] language", "Python is a programming language"),
        ("Neural networks have [MASK] layers", "Neural networks have hidden layers"),
        ("Trees reduce [MASK] pollution", "Trees reduce air pollution"),
        ("Robots perform [MASK] tasks", "Robots perform repetitive tasks"),
        ("Solar power is a [MASK] source", "Solar power is a renewable source"),
    ]
    vocab = build_vocab([s for s, _ in data] + [t for _, t in data])
    vocab_size = len(vocab)
    max_len = max(len(s.split()) for s, _ in data)
    model = FeedForwardMLM(vocab_size, 64, max_len)
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
            out = model(src_tensor)
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
        out = model(src_tensor)
        pred_ids = out.argmax(-1)
        print(f"Input: {src}")
        print(f"Output: {decode_batch(pred_ids.tolist(), vocab)}\n")

if __name__ == "__main__":
    main()
