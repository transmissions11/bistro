import torch

import lightning as L

from pathlib import Path

from lit_gpt.utils import lazy_load
from lit_gpt import Config, Tokenizer
from lit_gpt.utils import chunked_cross_entropy

from model import GPT
from sample import sample_model

from utils.padding import strip_right_pad
from utils.tensors import find_subtensor_end
from utils.vicuna import VICUNA_END_OF_USER_PROMPT_SEQUENCE
from utils.params import mark_only_soft_prompt_as_trainable


class LitModel(L.LightningModule):
    def __init__(
        self,
        model_config: Config,
        checkpoint_path: Path,
        learning_rate: float,
        warmup_ratio: float,
        min_lr_ratio: float,
        weight_decay: float,
        tokens_to_sample: int,
        tokenizer: Tokenizer,
        num_soft_prompt_tkns: int,
        soft_prompt_tkn: str,
    ):
        super().__init__()

        # Disable logging hyperparams, since we do it manually in fit.py.
        # TODO: We're currently saving the tokenizer as a hyperparam, which feels wrong.
        self.save_hyperparameters(ignore=["model"], logger=False)

    def configure_model(self):
        if self.model is not None:
            return

        self.model = GPT(
            config=self.hparams.model_config,
            soft_prompt_tkn=self.hparams.tokenizer.token_to_id(
                self.hparams.soft_prompt_tkn
            ),
            num_soft_prompt_tkns=self.hparams.num_soft_prompt_tkns,
        )

        with lazy_load(self.hparams.checkpoint_path) as checkpoint:
            self.model.load_state_dict(checkpoint, strict=False)

        mark_only_soft_prompt_as_trainable(self.model)

    def setup(self, stage):
        self.print(f"Resetting model caches for {stage}...\n")
        self.model.reset_caches()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch: dict, batch_idx):
        input_ids, targets = batch["input_ids"], batch["targets"]
        loss = self.compute_loss(input_ids, targets)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch: dict, batch_idx):
        input_ids, targets = batch["input_ids"], batch["targets"]
        loss = self.compute_loss(input_ids, targets)

        self.log(
            "val_loss",
            # We need to cast to float64 as types like bfloat16
            # have very low precision with larger values (~256+)
            # that results in inaccurate accumulation w/ on_epoch.
            loss.to(torch.float64),
            on_epoch=True,
            prog_bar=True,
        )

        if batch_idx < 10:
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

            self.print(f"\nInput: '{tokenizer.decode(sample[:prompt_end_idx + 1])}'")
            output = sample_model(
                self.model,
                idx=sample[: prompt_end_idx + 1],
                temperature=0.0,  # Sample greedily.
                max_new_tokens=self.hparams.tokens_to_sample,
            )[-self.hparams.tokens_to_sample :]
            self.print(f"Output: '{tokenizer.decode(output)}'")
            self.print(f"Target: '{tokenizer.decode(target[target != -1])}'\n")

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
            cycle_momentum=False,
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
