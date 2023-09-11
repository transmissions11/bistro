import lightning as L

import torch

from model import GPT

from lit_gpt.utils import (
    chunked_cross_entropy,
)

from lightning.pytorch.utilities import grad_norm

micro_batch_size = 1  # TODO: Set a larger value for this.

devices = 1

gradient_accumulation_iters = 3

epoch_size = 10_000_000  # TODO: Set this based on the actual dataset dynamically.
num_epochs = 4

max_iters = num_epochs * (epoch_size // micro_batch_size) // devices
# weight_decay = 0.02  # TODO: Should we be using this for finetuning?
warmup_steps = 2000

# TODO: Better config system.
# TODO: Cosine learning rate scheduler.
learning_rate = 3e-2
min_learning_rate = 0
weight_decay = 0.02


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

        # TODO: OHHH MAYBE ITS CUZ WEIGHT DECAY SHOULD BE 0.02 NOT 1e-2

        logits = self.model(input_ids)
        loss = chunked_cross_entropy(logits, targets, chunk_size=0)

        # TODO: do i need to do loss.item
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "samples", batch_idx, on_step=True, on_epoch=True, prog_bar=True
        )  # TODO: hack

        return loss

    def validation_step(self, batch: dict, batch_idx):
        input_ids, targets = batch["input_ids"], batch["targets"]

        logits = self.model(input_ids)
        loss = chunked_cross_entropy(logits, targets, chunk_size=0)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # TODO: How do we do linear warmup?
        # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LambdaLR.html
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            self.trainer.estimated_stepping_batches,
            eta_min=min_learning_rate,
            verbose=True,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            },
        }
