import torch

import lightning as L

from model import GPT

from lit_gpt.utils import chunked_cross_entropy


class LitModel(L.LightningModule):
    def __init__(
        self,
        model: GPT,
        learning_rate: float,
        min_learning_rate: float,  # TODO: or should this be %, vicuna does 0.03, anton does 0.05 (https://github.com/huggingface/transformers/pull/10229)
        warmup_steps: int,
        weight_decay: float,  # TODO: Should we be using this for finetuning?
    ):
        super().__init__()

        self.save_hyperparameters()

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

        # TODO: How do we do linear warmup?
        # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LambdaLR.html
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            self.trainer.estimated_stepping_batches,
            eta_min=self.hparams.min_learning_rate,
            verbose=True,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            },
        }
