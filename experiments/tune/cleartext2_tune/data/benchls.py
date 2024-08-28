import os

import pytorch_lightning as pl
import torch
from cleartext2_tune.data._utils import collate_fn, load_benchls, mask_sentence
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer


class BenchLSDataset(Dataset):
    def __init__(self, data, tokenizer, mask: bool = False):
        self._data = data
        self._tokenizer = tokenizer
        self._mask = mask

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        sent, word, pos, targets = self._data[idx]
        sent = mask_sentence(sent, word, pos)
        inp = self._tokenizer(
            sent,
            padding="max_length",
            truncation="do_not_truncate",  # could change position of mask token and hasn't been needed
            return_tensors="pt",
        )
        pos = torch.where(inp["input_ids"] == self._tokenizer.mask_token_id)[1]
        if not self._mask:
            inp["input_ids"][0][pos] = self._tokenizer.convert_tokens_to_ids(word)
        inp["positions"] = pos

        # TODO: use ranks to weight targets
        targets = [
            self._tokenizer(sub, add_special_tokens=False, return_tensors="pt")["input_ids"] for _, sub in targets
        ]
        targets = torch.tensor([t[0][0] for t in targets if t.numel() == 1])
        if targets.numel() == 0:
            inp["targets"] = torch.zeros(self._tokenizer.vocab_size)
        else:
            inp["targets"] = nn.functional.one_hot(targets, self._tokenizer.vocab_size).sum(dim=0)  # multi-hot encoding

        return inp


class BenchLSDataModule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        path: str,
        train_val_split: float,
        batch_size: int,
        num_workers: int = -1,
        augment: bool = True,
        mask: bool = False,
    ):
        super().__init__()
        self._tokenizer = tokenizer
        self._train_val_split = train_val_split
        self._path = path
        self._batch_size = batch_size
        self._num_workers = num_workers if num_workers > 0 else os.cpu_count()
        self._augment = augment
        self._mask = mask

    def setup(self, stage: str):
        train, val = load_benchls(self._path, self._train_val_split, augment=self._augment)
        self.train = BenchLSDataset(train, self._tokenizer, mask=self._mask)
        self.val = BenchLSDataset(val, self._tokenizer, mask=self._mask)

    def train_dataloader(self):
        return DataLoader(
            self.train, batch_size=self._batch_size, shuffle=True, collate_fn=collate_fn, num_workers=self._num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val, batch_size=self._batch_size, shuffle=False, collate_fn=collate_fn, num_workers=self._num_workers
        )
