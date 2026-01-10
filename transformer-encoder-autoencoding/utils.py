def build_vocab(sentences):
    vocab = {'[PAD]': 0, '[MASK]': 1}
    idx = 2
    for sent in sentences:
        for word in sent.split():
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    return vocab

def encode_batch(sentences, vocab, max_len):
    batch = []
    for sent in sentences:
        ids = [vocab.get(w, vocab['[MASK]']) for w in sent.split()]
        ids += [vocab['[PAD]']] * (max_len - len(ids))
        batch.append(ids)
    return batch

def decode_batch(batch_ids, vocab):
    inv_vocab = {v: k for k, v in vocab.items()}
    sentences = []
    for ids in batch_ids:
        words = [inv_vocab.get(i, '[UNK]') for i in ids if inv_vocab.get(i, '[UNK]') not in ['[PAD]']]
        sentences.append(' '.join(words))
    return sentences
