import time

import torch
from torch.nn import functional as F

import lightning as L

from pathlib import Path

from typing import Callable, Optional, cast

from lit_gpt import Config, Tokenizer

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
        freeze_criteria: Callable[[str], bool] = None,
    ):
        super().__init__()

        self.model = None  # This will get set in configure_model.

        # Have to assign manually as it's not pickle-able
        # and thus can't be saved via save_hyperparameters.
        self.freeze_criteria = freeze_criteria

        # Note logger=False since we already do it manually in fit.py.
        self.save_hyperparameters(ignore=["freeze_criteria"], logger=False)

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
                    device=self.device,  # TODO: idt these need to be on device manually anymore
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

        # Can't use self.print here since it will try
        # to print via the not-yet-existent prog bar.
        def g0_print(msg: str) -> None:
            if self.trainer.is_global_zero:
                print(msg)

            return time.time()  # For timing convenience.

        t0 = g0_print("Initializing model...")
        self.model = GPT(
            config=self.hparams.model_config,
            soft_prompt_tkn=self.hparams.tokenizer.token_to_id(
                self.hparams.soft_prompt_tkn
            ),
            num_soft_prompt_tkns=self.hparams.num_soft_prompt_tkns,
        )
        g0_print(f"Initialized model in {time.time() - t0:.3f}s.")

        if self.hparams.checkpoint_path is not None:
            t0 = g0_print(
                f"Loading checkpoint weights from {self.hparams.checkpoint_path}..."
            )
            self.model.load_state_dict(
                torch.load(str(self.hparams.checkpoint_path), mmap=True),
                strict=False,
                assign=True,
            )
            g0_print(f"Loaded checkpoint in {time.time() - t0:.3f}s.")

        if self.freeze_criteria is not None:
            t0 = g0_print("Freezing specified parameters...")
            freeze_parameters(self.model, self.freeze_criteria)
            g0_print(f"Froze specified parameters in {time.time() - t0:.3f}s.")

        g0_print("Watching model gradients with W&B...")
        cast(WandbLogger, self.trainer.logger).watch(self.model)

        g0_print("Done loading & configuring model.")

    def on_train_start(self) -> None:
        self.print(f"Resetting model caches for training...\n")
        self.model.reset_caches()

    def compute_loss(self, input_ids, targets):
        logits = self.model(input_ids)
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
        )
