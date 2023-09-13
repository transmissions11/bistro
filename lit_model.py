from lit_gpt import Tokenizer
import torch

import lightning as L

from model import GPT

from lit_gpt.utils import chunked_cross_entropy
from sample import sample_model

from utils.padding import strip_right_pad
from utils.tensors import find_subtensor_end
from utils.vicuna import VICUNA_END_OF_USER_PROMPT_SEQUENCE
from lightning.pytorch.callbacks import ModelCheckpoint


class LitModel(L.LightningModule):
    def __init__(
        self,
        model: GPT,
        learning_rate: float,
        warmup_ratio: float,
        min_lr_ratio: float,
        weight_decay: float,
        tokens_to_sample: int,
        tokenizer: Tokenizer,
    ):
        super().__init__()

        # Disable logging hyperparams, since we do it manually in fit.py.
        self.save_hyperparameters(ignore=["model"], logger=False)

        self.model = model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch: dict, batch_idx):
        input_ids, targets = batch["input_ids"], batch["targets"]
        loss = self.compute_loss(input_ids, targets)

        self.log("train/loss", loss)

        return loss

    def validation_step(self, batch: dict, batch_idx):
        input_ids, targets = batch["input_ids"], batch["targets"]
        loss = self.compute_loss(input_ids, targets)

        # Disabling on_epoch until I can figure out
        # why mean & sum reduction give weird results.
        self.log("val/loss", loss, on_step=True, on_epoch=False)

        tokenizer = self.hparams.tokenizer

        sample = strip_right_pad(input_ids[0])
        target = strip_right_pad(targets[0])

        prompt_end_idx = find_subtensor_end(
            sample,
            tokenizer.encode(
                VICUNA_END_OF_USER_PROMPT_SEQUENCE,
                device=self.device,
            ),
        )

        print(f"Input: {tokenizer.decode(sample[:prompt_end_idx + 1])}")
        output = sample_model(
            self.model,
            idx=sample[: prompt_end_idx + 1],
            temperature=0.00,  # Sample greedily.
            max_new_tokens=self.hparams.tokens_to_sample,
        )[-self.hparams.tokens_to_sample :]
        print(f"Output:", tokenizer.decode(output))
        print(f"Target:", tokenizer.decode(target[target != -1]))
        print("\n\n")

        return loss

    def compute_loss(self, input_ids, targets):
        logits = self.model(input_ids)
        return chunked_cross_entropy(logits, targets, chunk_size=0)

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
