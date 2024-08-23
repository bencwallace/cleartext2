import os

import hydra
import pytorch_lightning as pl
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    DistilBertModel,
    DistilBertPreTrainedModel,
    PreTrainedTokenizer,
)

# TODO: check if fast tokenizers can be used
os.environ["TOKENIZERS_PARALLELISM"] = "false"
wandb.require("core")


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


def load_benchls(path, augment=False):
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

    if augment:
        data = _augment_benchls(data)
    return data


class BenchLSDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, path: str, augment: bool = True):
        data = load_benchls(path, augment=augment)
        self._data = []
        for sent, word, pos, targets in data:
            # surround complex word with [SEP] tokens
            # here we tokenize with str.split() for consistency with the BenchLS data
            sent_toks = sent.lower().split()
            word_toks = word.split()
            # note that BenchLS sentences only contain a single, final [SEP] token
            sent_toks[pos : pos + len(word_toks)] = ["[SEP]"] + word_toks + ["[SEP]"]
            sent = " ".join(sent_toks)

            # TODO: use ranks to weight targets
            targets = [tokenizer(sub, add_special_tokens=False, return_tensors="pt")["input_ids"] for _, sub in targets]
            targets = torch.tensor([t[0][0] for t in targets if t.numel() == 1])
            if targets.numel():
                # for now only consider single token substitutions
                self._data.append((sent, targets))

        self._tokenizer = tokenizer

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        sent, targets = self._data[idx]
        # TODO: consider tokenizing in __init__
        inp = self._tokenizer(
            sent,
            padding="max_length",
            truncation="do_not_truncate",  # could change position of mask token and hasn't been needed
            return_tensors="pt",
        )

        inp["targets"] = nn.functional.one_hot(targets, self._tokenizer.vocab_size).sum(dim=0)  # multi-hot encoding

        return inp


def collate_fn(batch):
    return {
        "input_ids": torch.vstack([ex["input_ids"] for ex in batch]),
        "attention_mask": torch.vstack([ex["attention_mask"] for ex in batch]),
        "targets": torch.stack([ex["targets"] for ex in batch]),
    }


class DistilBertForLexicalSimplification(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.distilbert = DistilBertModel(config)
        self.dropout = nn.Dropout(config.seq_classif_dropout)
        self.classifier = nn.Linear(config.dim, config.vocab_size)
        self.post_init()

    def forward(self, input_ids, attention_mask=None):
        outputs = self.distilbert(input_ids, attention_mask=attention_mask)
        hidden_state = outputs[0]
        pooled_output = hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class LexicalSimplificationModule(pl.LightningModule):
    def __init__(self, model_name="distilbert-base-uncased"):
        super().__init__()

        self.model = DistilBertForLexicalSimplification.from_pretrained(model_name)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids, attention_mask=attention_mask)

    def training_step(self, batch, _):
        output = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        loss = self.loss_fn(output, batch["targets"].float())
        self.log("train/loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)


@hydra.main(version_base=None, config_path="conf", config_name="tune")
def main(cfg: DictConfig):
    ckpt_callback = ModelCheckpoint(save_top_k=0, save_last=False)
    logger = WandbLogger(project=cfg.project, mode=cfg.wandb_mode)
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[ckpt_callback],
        max_epochs=cfg.max_epochs,
        limit_train_batches=cfg.limit_train_batches,
        log_every_n_steps=cfg.log_every_n_steps,
    )

    module = LexicalSimplificationModule(cfg.model_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, clean_up_tokenization_spaces=True)
    dataset = instantiate(cfg.dataset, tokenizer=tokenizer)
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=cfg.num_workers
    )
    trainer.fit(module, loader)


if __name__ == "__main__":
    main()
