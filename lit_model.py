import lightning as L

import torch

from model import GPT

from lit_gpt.utils import (
    chunked_cross_entropy,
)

micro_batch_size = 1  # TODO: Set a larger value for this.

devices = 1

gradient_accumulation_iters = 3

epoch_size = 10_000_000  # TODO: Set this based on the actual dataset dynamically.
num_epochs = 4

max_iters = num_epochs * (epoch_size // micro_batch_size) // devices
weight_decay = 0.02  # TODO: Should we be using this for finetuning?
warmup_steps = (
    2 * (epoch_size // micro_batch_size) // devices // gradient_accumulation_iters
)

# TODO: Better config system.
# TODO: Cosine learning rate scheduler.
learning_rate = 1
weight_decay = 1e-2


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

        print(batch_idx)

        if batch_idx <= warmup_steps:
            lr = learning_rate * batch_idx / warmup_steps
            for param_group in self.optimizers()[0].param_groups:
                param_group["lr"] = lr

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
        return torch.optim.AdamW(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
