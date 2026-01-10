import torch


import os
def save_model(model, vocab, path="results/model_wiki.pth"):
    # Ensure absolute path and directory exists
    abs_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab
    }, abs_path)
    print(f"Model saved to {abs_path}")

def load_model(model_class, path="results/model_wiki.pth", **model_kwargs):
    abs_path = os.path.abspath(path)
    checkpoint = torch.load(abs_path, map_location='cpu')
    model = model_class(**model_kwargs)
    model.load_state_dict(checkpoint['model_state_dict'])
    vocab = checkpoint['vocab']
    return model, vocab
