import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import torch

from encoder import TransformerEncoder
from synthetic_samples import MLM_PAIRS
from utils import build_vocab, encode_batch


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def generate_architecture_diagram(out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.set_axis_off()

    boxes = [
        (0.02, 0.35, 0.15, 0.3, "Token\nIDs"),
        (0.20, 0.35, 0.18, 0.3, "Embedding"),
        (0.42, 0.35, 0.22, 0.3, "Positional\nEncoding"),
        (0.68, 0.35, 0.26, 0.3, "Encoder Layers\n(Self-Attn + FFN)"),
    ]

    for x, y, w, h, label in boxes:
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1.5,
            edgecolor="black",
            facecolor="#E6F0FF",
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=11)

    ax.annotate("", xy=(0.20, 0.5), xytext=(0.17, 0.5), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(0.42, 0.5), xytext=(0.38, 0.5), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(0.68, 0.5), xytext=(0.64, 0.5), arrowprops=dict(arrowstyle="->", lw=2))

    ax.text(0.86, 0.18, "MLM Head → vocab logits\nCLS Head → class logits", ha="center", fontsize=11)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def generate_attention_heatmap(out_path: str) -> None:
    # Use the small lab dataset so this is deterministic and quick
    pairs = MLM_PAIRS

    sentences = [s for s, _ in pairs] + [t for _, t in pairs]
    vocab = build_vocab(sentences)
    max_len = max(len(s.split()) for s, _ in pairs)

    model = TransformerEncoder(len(vocab), 64, 4, 2, 128, max_len)
    model.eval()

    sent = pairs[0][0]
    x = torch.tensor(encode_batch([sent], vocab, max_len))

    with torch.no_grad():
        _, attn = model(x)

    # attn: [B, heads, T, T]; pick head 0
    matrix = attn[0][0].cpu().numpy()

    tokens = sent.split() + ["[PAD]"] * (max_len - len(sent.split()))

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(matrix, cmap="viridis", ax=ax)
    ax.set_title("Attention Heatmap (Head 0)")
    ax.set_xlabel("Key tokens")
    ax.set_ylabel("Query tokens")

    # Label only non-pad tokens to keep it readable
    ax.set_xticks([i + 0.5 for i in range(len(tokens))])
    ax.set_yticks([i + 0.5 for i in range(len(tokens))])
    ax.set_xticklabels(tokens, rotation=45, ha="right")
    ax.set_yticklabels(tokens, rotation=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    ensure_dir(results_dir)

    generate_architecture_diagram(os.path.join(results_dir, "encoder_architecture.png"))
    generate_attention_heatmap(os.path.join(results_dir, "attention_heatmap.png"))

    print("Generated: results/encoder_architecture.png")
    print("Generated: results/attention_heatmap.png")


if __name__ == "__main__":
    main()
