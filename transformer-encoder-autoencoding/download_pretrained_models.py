"""Download/cache all pre-trained MLM models used by the Streamlit app.

Run:
  python transformer-encoder-autoencoding/download_pretrained_models.py

This will download model weights into your local Hugging Face cache so the UI loads faster.
"""

from transformers import pipeline


def main() -> None:
    models = [
        "bert-base-uncased",
        "distilbert-base-uncased",
        "roberta-base",
        "bert-large-uncased",
    ]

    for model_id in models:
        print(f"\n=== Downloading/caching: {model_id} ===")
        nlp = pipeline("fill-mask", model=model_id)
        mask = nlp.tokenizer.mask_token
        sample = f"Transformers use {mask} attention"
        top = nlp(sample, top_k=1)[0]
        print(
            f"OK: {model_id} | mask={mask} | sample_pred={top['sequence']} | score={top['score']:.4f}"
        )

    print("\nAll selected models are cached.")


if __name__ == "__main__":
    main()
