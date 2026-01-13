import os
import torch


def _resolve_path(path: str) -> str:
    # If a relative path is provided, resolve it relative to this module's folder,
    # not the current working directory.
    if os.path.isabs(path):
        return path
    base_dir = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(base_dir, path))


def save_model(model, vocab, path: str = "results/model_wiki.pth", config: dict | None = None):
    abs_path = _resolve_path(path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab": vocab,
        "config": config or {},
    }, abs_path)
    print(f"Model saved to {abs_path}")


def load_model(model_class, path: str = "results/model_wiki.pth", **model_kwargs):
    abs_path = _resolve_path(path)
    checkpoint = torch.load(abs_path, map_location="cpu")
    model = model_class(**model_kwargs)
    model.load_state_dict(checkpoint["model_state_dict"])
    vocab = checkpoint["vocab"]
    config = checkpoint.get("config", {})
    return model, vocab, config
