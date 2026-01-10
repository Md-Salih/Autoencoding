import torch
import torch.nn as nn
import torch.optim as optim

from encoder import TransformerEncoder
from utils import build_vocab, encode_batch


# Topic classification demo using the lab SAMPLE sentences.
# This satisfies: "Use the same encoder to classify sentences".
DATA = [
    ("Transformers use [MASK] attention", "AI"),
    ("Mars is called the [MASK] planet", "Space"),
    ("Online learning improves [MASK] access", "Education"),
    ("Exercise improves [MASK] health", "Health"),
    ("Cricket is a [MASK] sport", "Sports"),
    ("Python is a [MASK] language", "Computing"),
    ("Neural networks have [MASK] layers", "AI"),
    ("Trees reduce [MASK] pollution", "Environment"),
    ("Robots perform [MASK] tasks", "Robotics"),
    ("Solar power is a [MASK] source", "Energy"),
]


def main():
    sentences = [s for s, _ in DATA]
    topics = [t for _, t in DATA]

    label_names = sorted(set(topics))
    label_to_id = {name: i for i, name in enumerate(label_names)}
    y = torch.tensor([label_to_id[t] for t in topics], dtype=torch.long)

    vocab = build_vocab(sentences)
    if "[CLS]" not in vocab:
        vocab["[CLS]"] = len(vocab)

    max_len = max(len(s.split()) for s in sentences) + 1  # +1 for [CLS]
    x = torch.tensor(encode_batch(sentences, vocab, max_len, add_cls=True), dtype=torch.long)

    model = TransformerEncoder(
        vocab_size=len(vocab),
        d_model=64,
        n_heads=4,
        num_layers=2,
        d_ff=128,
        max_len=max_len,
        num_classes=len(label_names),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(100):
        logits, _ = model(x, task="cls")
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            preds = logits.argmax(dim=-1)
            acc = (preds == y).float().mean().item()
            print(f"Epoch {epoch+1:03d} | loss={loss.item():.4f} | acc={acc:.2%}")

    model.eval()
    with torch.no_grad():
        logits, _ = model(x, task="cls")
        preds = logits.argmax(dim=-1).tolist()

    print("\nPredictions on training examples:")
    for s, pred_id in zip(sentences, preds):
        print(f"- {s} -> {label_names[pred_id]}")


if __name__ == "__main__":
    main()
