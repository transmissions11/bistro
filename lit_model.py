import time
import torch

import lightning as L

from pathlib import Path

from typing import Optional, cast

from lit_gpt import Config, Tokenizer
from lit_gpt.utils import chunked_cross_entropy

from lightning.pytorch.loggers import WandbLogger

from model import GPT
from sample import sample_model

from utils.padding import strip_right_pad
from utils.params import freeze_parameters
from utils.tensors import find_subtensor_end
from utils.vicuna import VICUNA_END_OF_USER_PROMPT_SEQUENCE


class LitModel(L.LightningModule):
    def __init__(
        self,
        model_config: Config,
        tokenizer: Tokenizer,
        ###############################
        learning_rate: float,
        warmup_ratio: float,
        min_lr_ratio: float,
        ###############################
        weight_decay: float,
        ###############################
        tokens_to_sample: int,
        ###############################
        num_soft_prompt_tkns: int,
        soft_prompt_tkn: str,
        ###############################
        checkpoint_path: Optional[Path] = None,
    ):
        super().__init__()

        self.model = None  # This will get set in configure_model.

        # Disable logging hyperparams, since we do it manually in fit.py.
        # TODO: We're currently saving the tokenizer as a hyperparam, which feels wrong.
        self.save_hyperparameters(ignore=["model"], logger=False)

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
            # Need to upcast precision as types like bfloat16
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

    def configure_model(self):
        if self.model is not None:
            return

        # TODO: Only do on rank 0 (WHY DOENST SELF>PRINT WORK)
        print("Initializing GPT model...")
        self.model = GPT(
            config=self.hparams.model_config,
            soft_prompt_tkn=self.hparams.tokenizer.token_to_id(
                self.hparams.soft_prompt_tkn
            ),
            num_soft_prompt_tkns=self.hparams.num_soft_prompt_tkns,
        )

        # If a checkpoint path was provided, we'll
        # load its state dict in, with strict=False.
        if self.hparams.checkpoint_path is not None:
            print(f"Loading model weights from {self.hparams.checkpoint_path}...")
            t0 = time.time()
            self.model.load_state_dict(
                torch.load(str(self.hparams.checkpoint_path), mmap=True),
                strict=False,
                assign=False,  # TODO: Try assign=True
            )
            print(f"Loaded model weights in {time.time() - t0:.2f}s.")

        # TODO: Should we use self or self.model?
        print("Setting trainable parameters...")
        freeze_parameters(self.model, lambda name: "soft_prompt" not in name)

        # TODO: Should we use self or self.model?
        print("Watching model gradients with W&B...")
        cast(WandbLogger, self.trainer.logger).watch(self.model)

        print("Done loading & configuring model.")

    def on_train_start(self) -> None:
        self.print(f"Resetting model caches for training...\n")
        self.model.reset_caches()
