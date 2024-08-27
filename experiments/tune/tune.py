import os

import hydra
import pytorch_lightning as pl
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchmetrics.retrieval import RetrievalMAP, RetrievalPrecision, RetrievalRecall
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DistilBertModel,
    DistilBertPreTrainedModel,
    PreTrainedTokenizer,
)

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


def collate_fn(batch):
    return {
        "input_ids": torch.vstack([ex["input_ids"] for ex in batch]),
        "attention_mask": torch.vstack([ex["attention_mask"] for ex in batch]),
        "targets": torch.stack([ex["targets"] for ex in batch]),
        "positions": torch.cat([ex["positions"] for ex in batch]),
    }


class BenchLSDataModule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        path: str,
        train_val_split: float,
        batch_size: int,
        num_workers: int = -1,
        augment: bool = True,
    ):
        super().__init__()
        self._tokenizer = tokenizer
        self._train_val_split = train_val_split
        self._path = path
        self._batch_size = batch_size
        self._num_workers = num_workers if num_workers > 0 else os.cpu_count()
        self._augment = augment

    def setup(self, stage: str):
        train, val = load_benchls(self._path, self._train_val_split, augment=self._augment)
        self.train = BenchLSDataset(train, self._tokenizer)
        self.val = BenchLSDataset(val, self._tokenizer)

    def train_dataloader(self):
        return DataLoader(
            self.train, batch_size=self._batch_size, shuffle=True, collate_fn=collate_fn, num_workers=self._num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val, batch_size=self._batch_size, shuffle=False, collate_fn=collate_fn, num_workers=self._num_workers
        )


class TaggedLS(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.distilbert = DistilBertModel(config)
        self.dropout = nn.Dropout(config.seq_classif_dropout)
        self.classifier = nn.Linear(config.dim, config.vocab_size)
        self.post_init()

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.distilbert(input_ids, attention_mask=attention_mask)
        hidden_state = outputs[0]
        pooled_output = hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class LexicalSimplificationModule(pl.LightningModule):
    def __init__(self, model_name="distilbert-base-uncased", freeze: bool = True, lr=2e-5, top_k=10):
        super().__init__()

        self._lr = lr
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        if freeze:
            for param in self.model.distilbert.parameters():
                param.requires_grad = False
        self.loss_fn = nn.BCEWithLogitsLoss()

        self._top_k = top_k
        self.rmap = RetrievalMAP(top_k=top_k)
        self.rprec = RetrievalPrecision(top_k=top_k)
        self.rrec = RetrievalRecall(top_k=top_k)

    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids, attention_mask=attention_mask)

    def training_step(self, batch, _):
        output = self.model(batch["input_ids"], attention_mask=batch["attention_mask"])
        output = output.logits[torch.arange(output.logits.size(0)), batch["positions"]]
        loss = self.loss_fn(output, batch["targets"].float())
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, _):
        output = self.model(batch["input_ids"], attention_mask=batch["attention_mask"])
        output = output.logits[torch.arange(output.logits.size(0)), batch["positions"]]

        loss = self.loss_fn(output, batch["targets"].float())
        self.log("val/loss", loss)

        preds = torch.sigmoid(output).flatten()
        targets = batch["targets"].flatten()
        indexes = torch.arange(output.size(0)).repeat_interleave(self.model.config.vocab_size)

        self.rmap(preds, targets, indexes=indexes)
        self.rprec(preds, targets, indexes=indexes)
        self.rrec(preds, targets, indexes=indexes)
        self.log(f"val/rMAP@k={self._top_k}", self.rmap, on_step=False, on_epoch=True)
        self.log(f"val/rPrec@k={self._top_k}", self.rprec, on_step=False, on_epoch=True)
        self.log(f"val/rRec@k={self._top_k}", self.rrec, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self._lr)


@hydra.main(version_base=None, config_path="conf", config_name="tune")
def main(cfg: DictConfig):
    ckpt_callback = ModelCheckpoint(save_top_k=0, save_last=True)
    logger = WandbLogger(
        log_model=False, project=cfg.project, mode=cfg.wandb_mode, config=OmegaConf.to_container(cfg, resolve=True)
    )
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[ckpt_callback],
        max_epochs=cfg.max_epochs,
        limit_train_batches=cfg.limit_train_batches,
        limit_val_batches=cfg.limit_val_batches,
        log_every_n_steps=cfg.log_every_n_steps,
        overfit_batches=cfg.overfit_batches,
    )

    module = LexicalSimplificationModule(cfg.model_name, freeze=cfg.freeze, lr=cfg.lr, top_k=cfg.top_k)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, clean_up_tokenization_spaces=True)
    dm = instantiate(cfg.data, tokenizer=tokenizer)
    trainer.fit(module, dm)


if __name__ == "__main__":
    main()
