import lightning as L

import torch

from model import GPT

from lit_gpt.utils import (
    chunked_cross_entropy,
)


class LitModel(L.LightningModule, GPT):
    def __init__(
        self,
    ):
        super().__init__()
        # TODO: Try FusedCrossEntropyLoss from TinyLlama.
        self.loss_fn = chunked_cross_entropy

    def training_step(self, batch, batch_idx):
        input_ids, targets = batch

        logits = self(input_ids)
        loss = self.loss_fn(logits, targets)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, targets = batch

        logits = self(input_ids)
        loss = self.loss_fn(logits, targets)

        return loss

    def configure_optimizers(self):
        # TODO: Better config system.
        learning_rate = 1
        weight_decay = 1e-2

        return torch.optim.AdamW(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
