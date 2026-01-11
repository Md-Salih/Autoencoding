def build_vocab(sentences):
    # Keep special tokens minimal and stable.
    # NOTE: Some older checkpoints may not include newer tokens; code below stays backward-compatible.
    vocab = {'[PAD]': 0, '[MASK]': 1, '[UNK]': 2}
    idx = 3
    for sent in sentences:
        for word in sent.split():
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    return vocab

def encode_batch(sentences, vocab, max_len, add_cls: bool = False):
    batch = []
    for sent in sentences:
        pad_id = vocab.get('[PAD]', 0)
        unk_fallback = vocab.get('[UNK]', vocab.get('[MASK]', 0))

        tokens = sent.split()
        if add_cls:
            cls_id = vocab.get('[CLS]')
            if cls_id is None:
                raise ValueError("add_cls=True but vocab has no [CLS] token")
            ids = [cls_id] + [vocab.get(w, unk_fallback) for w in tokens]
        else:
            ids = [vocab.get(w, unk_fallback) for w in tokens]

        ids = ids[:max_len]
        ids += [pad_id] * (max_len - len(ids))
        batch.append(ids)
    return batch

def decode_batch(batch_ids, vocab):
    inv_vocab = {v: k for k, v in vocab.items()}
    sentences = []
    for ids in batch_ids:
        words = [inv_vocab.get(i, '[UNK]') for i in ids if inv_vocab.get(i, '[UNK]') not in ['[PAD]']]
        sentences.append(' '.join(words))
    return sentences
