import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics.retrieval import RetrievalMAP, RetrievalPrecision, RetrievalRecall
from transformers import AutoModelForMaskedLM


class LSModule(pl.LightningModule):
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
