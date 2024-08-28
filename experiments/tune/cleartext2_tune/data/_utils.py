import torch
from torch.utils.data import random_split


def _augment_benchls(data):
    augmented = []
    for sent, word, pos, targets in data:
        augmented.append((sent, word, pos, targets))

        # use the fact that targets are sorted by increasing rank
        toks = sent.lower().split()
        for i, (rank, sub) in enumerate(targets):
            candidates = [j for j in range(i) if targets[j][0] < rank]
            if candidates:
                stop = max(candidates)
                toks[pos] = sub
                sent = " ".join(toks)
                augmented.append((sent, sub, pos, targets[: stop + 1]))

    return augmented


def load_benchls(path, train_val_split: float, augment=False):
    with open(path) as f:
        lines = f.readlines()

    data = []
    for line in lines:
        sent, word, pos, *targets = line.split("\t")
        for i, t in enumerate(targets):
            rank, sub = t.split(":")
            sub = sub.strip()
            targets[i] = (int(rank), sub)
        data.append((sent, word.strip(), int(pos), targets))

    # important to split prior to augmenting
    train_size = int(train_val_split * len(data))
    val_size = len(data) - train_size
    train, val = random_split(data, [train_size, val_size])
    if augment:
        train = _augment_benchls(train)
        val = _augment_benchls(val)
    return train, val


def mask_sentence(sent, word, pos):
    # replace complex word with [MASK] token
    sent_toks = sent.lower().split()
    word_toks = word.split()
    sent_toks[pos : pos + len(word_toks)] = ["[MASK]"]
    return " ".join(sent_toks)


def collate_fn(batch):
    return {
        "input_ids": torch.vstack([ex["input_ids"] for ex in batch]),
        "attention_mask": torch.vstack([ex["attention_mask"] for ex in batch]),
        "targets": torch.stack([ex["targets"] for ex in batch]),
        "positions": torch.cat([ex["positions"] for ex in batch]),
    }
