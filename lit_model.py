import lightning as L

import torch

from model import GPT


class LitModel(L.LightningModule):
    def __init__(
        self,
        model: GPT,
    ):
        super().__init__()
        self.model = model
        # TODO: Try FusedCrossEntropyLoss from TinyLlama.
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)

    def training_step(self, batch, batch_idx):
        input_ids, targets = batch

        logits = self.model(input_ids)
        loss = self.loss_fn(logits, targets)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, targets = batch

        logits = self.model(input_ids)
        loss = self.loss_fn(logits, targets)

        return loss

    def configure_optimizers(self):
        # TODO: Better config system.
        learning_rate = 1
        weight_decay = 1e-2

        return torch.optim.AdamW(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
