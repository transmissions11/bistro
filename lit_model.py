import torch

import lightning as L

from model import GPT

from lit_gpt.utils import chunked_cross_entropy


class LitModel(L.LightningModule):
    def __init__(
        self,
        model: GPT,
        learning_rate: float,
        warmup_ratio: float,
        min_lr_ratio: float,
        weight_decay: float,
    ):
        super().__init__()

        # Disable logging hyperparams, since we do it manually in fit.py.
        self.save_hyperparameters(ignore=["model"], logger=False)

        self.model = model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch: dict, batch_idx):
        input_ids, targets = batch["input_ids"], batch["targets"]

        logits = self.model(input_ids)
        loss = chunked_cross_entropy(logits, targets, chunk_size=0)

        self.log("train/loss", loss, on_step=True)

        return loss

    def validation_step(self, batch: dict, batch_idx):
        input_ids, targets = batch["input_ids"], batch["targets"]

        logits = self.model(input_ids)
        loss = chunked_cross_entropy(logits, targets, chunk_size=0)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=self.hparams.warmup_ratio,
            cycle_momentum=False,  #
            div_factor=1e10,  # Large number, so we start at 0.
            final_div_factor=1 / (self.hparams.min_lr_ratio + 1e-10),
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            },
        }
