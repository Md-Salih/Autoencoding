import torch
import torch.nn as nn
import torch.optim as optim

from encoder import TransformerEncoder
from save_load_model import save_model
from synthetic_samples import MLM_PAIRS
from utils import build_vocab, encode_batch, decode_batch

DATA = MLM_PAIRS


def main():
    vocab = build_vocab([s for s, _ in DATA] + [t for _, t in DATA])
    vocab_size = len(vocab)
    max_len = max(len(s.split()) for s, _ in DATA)

    # Model config (small + fast)
    config = {
        "vocab_size": vocab_size,
        "d_model": 64,
        "n_heads": 4,
        "num_layers": 2,
        "d_ff": 128,
        "max_len": max_len,
        "task": "mlm",
    }

    model = TransformerEncoder(
        vocab_size=vocab_size,
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        num_layers=config["num_layers"],
        d_ff=config["d_ff"],
        max_len=config["max_len"],
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 100
    for epoch in range(epochs):
        total_loss = 0.0
        for src, tgt in DATA:
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

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(DATA):.4f}")

    # Save checkpoint for the Streamlit app
    save_model(model, vocab, path="results/model_lab.pth", config=config)

    # Quick test
    for src, _ in DATA:
        src_ids = encode_batch([src], vocab, max_len)
        src_tensor = torch.tensor(src_ids)
        out, _ = model(src_tensor)
        pred_ids = out.argmax(-1)
        print(f"Input: {src}")
        print(f"Output: {decode_batch(pred_ids.tolist(), vocab)[0]}\n")


if __name__ == "__main__":
    main()
