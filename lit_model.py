import torch

import lightning as L

from model import GPT

from lit_gpt.utils import chunked_cross_entropy

import math

# TODO: Try https://pytorch-lightning.readthedocs.io/en/1.4.9/advanced/lr_finder.html
# INSPO from https://github.com/the-full-stack/fsdl-text-recognizer-2022/blob/9d6bc110822761398e03eadb978af793c3c40bc1/text_recognizer/lit_models/transformer.py#L22-L42


class LitModel(L.LightningModule):
    def __init__(
        self,
        model: GPT,
        learning_rate: float,
        warmup_ratio: float,
        min_lr_ratio: float,
        weight_decay: float,  # TODO: Should we be using this for finetuning?
    ):
        super().__init__()

        self.save_hyperparameters(
            ignore=["model"]
        )  # TODO: How to log grad accum steps and micro batch size to wandb?

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
