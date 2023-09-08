import lightning as L

import torch

from model import GPT

from lit_gpt.utils import (
    chunked_cross_entropy,
)


class LitModel(L.LightningModule):
    def __init__(
        self,
        model: GPT,
    ):
        super().__init__()
        self.model = model

        # TODO: Try FusedCrossEntropyLoss from TinyLlama.
        # Also, anything other than chunk_size=0 is broken.

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch: dict, batch_idx):
        input_ids, targets = batch["input_ids"], batch["targets"]

        logits = self.model(input_ids)
        loss = chunked_cross_entropy(logits, targets, chunk_size=0)

        # TODO: do i need to do loss.item
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: dict, batch_idx):
        input_ids, targets = batch["input_ids"], batch["targets"]

        logits = self.model(input_ids)
        loss = chunked_cross_entropy(logits, targets, chunk_size=0)

        return loss

    def configure_optimizers(self):
        # TODO: Better config system.
        # TODO: Cosine learning rate scheduler.
        learning_rate = 0.3
        weight_decay = 1e-2

        return torch.optim.AdamW(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
