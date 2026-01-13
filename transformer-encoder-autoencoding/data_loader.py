from datasets import load_dataset
import random

def get_wiki_sentences(num_samples=1000, min_len=6, max_len=20):
    wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
    sentences = []
    for article in wiki:
        text = article.get('text', '')
        for line in text.split('\n'):
            line = line.strip()
            if min_len <= len(line.split()) <= max_len:
                sentences.append(line)
        if len(sentences) >= num_samples:
            break
    random.shuffle(sentences)
    return sentences[:num_samples]

if __name__ == "__main__":
    sents = get_wiki_sentences(10)
    for s in sents:
        print(s)
