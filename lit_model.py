import time
import torch

from torch.nn import functional as F

import lightning as L

from pathlib import Path

from typing import Callable, Optional, cast

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import grad_norm

from lit_gpt import Config, Tokenizer

from utils.padding import strip_right_pad
from utils.tensors import find_subtensor_end
from utils.vicuna import VICUNA_END_OF_USER_PROMPT_SEQUENCE
from utils.params import freeze_parameters, init_weights_optimally

from model import GPT
from sample import sample_model


class LitModel(L.LightningModule):
    def __init__(
        self,
        model_config: Config,
        tokenizer: Tokenizer,
        ###############################
        learning_rate: float,
        warmup_ratio: float,
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

        # Assign these manually as they don't pickle well
        # or shouldn't be saved via save_hyperparameters.
        self.freeze_criteria = freeze_criteria
        self.checkpoint_path = checkpoint_path
        self.tokenizer = tokenizer

        # Note: logger=False since we already log hparams it manually in fit.py.
        self.save_hyperparameters(
            ignore=["freeze_criteria", "checkpoint_path", "tokenizer"], logger=False
        )

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self.layer, norm_type=2)
        self.log_dict(norms)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        inputs, targets = batch["inputs"], batch["targets"]
        loss = self.compute_loss(inputs, targets)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        inputs, targets = batch["inputs"], batch["targets"]
        loss = self.compute_loss(inputs, targets)

        self.log(
            "val_loss",
            # Need to upcast precision as types like bfloat16
            # have very low precision with larger values (~256+)
            # that results in inaccurate accumulation w/ on_epoch.
            # https://github.com/Lightning-AI/lightning/issues/18620
            loss.to(torch.float64),
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        if batch_idx == 0:
            tokenizer = self.tokenizer

            inputs_list = []
            outputs_list = []
            targets_list = []

            for i in range(len(inputs)):
                sample = strip_right_pad(inputs[i])
                target = strip_right_pad(targets[i])

                prompt_end_idx = find_subtensor_end(
                    sample,
                    tokenizer.encode(
                        VICUNA_END_OF_USER_PROMPT_SEQUENCE, device=self.device
                    ),
                )

                input_sample = sample[: prompt_end_idx + 1]
                inputs_list.append(tokenizer.decode(input_sample))

                output = sample_model(
                    self.model,
                    idx=input_sample,
                    temperature=0.00,  # Sample greedily.
                    max_new_tokens=self.hparams.tokens_to_sample,
                )[-self.hparams.tokens_to_sample :]
                outputs_list.append(tokenizer.decode(output))
                # Note: This strips away ignored_tkn tokens entirely, which may
                # lead to confusion if ignored_tkns are used between real tokens.
                targets_list.append(tokenizer.decode(target[target != -1]))

            columns = ["input", "output", "target"]
            data = list(zip(inputs_list, outputs_list, targets_list))
            self.logger.log_text(key="my_samples", columns=columns, data=data)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.learning_rate,
            pct_start=self.hparams.warmup_ratio,
            total_steps=self.trainer.estimated_stepping_batches,
            div_factor=1e10,  # Large number, so we start at 0.
            cycle_momentum=False,
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
            soft_prompt_tkn=self.tokenizer.token_to_id(self.hparams.soft_prompt_tkn),
            num_soft_prompt_tkns=self.hparams.num_soft_prompt_tkns,
        )
        g0_print(f"Initialized model in {time.time() - t0:.3f}s.")

        t0 = g0_print("Initializing weights with optimal randomness...")
        self.model.apply(init_weights_optimally)
        g0_print(f"Initialized weights optimally in {time.time() - t0:.3f}s.")

        if self.checkpoint_path is not None:
            t0 = g0_print(f"Loading checkpoint weights from {self.checkpoint_path}...")
            self.model.load_state_dict(
                torch.load(str(self.checkpoint_path), mmap=True),
                strict=False,
                assign=True,
            )
            g0_print(f"Loaded checkpoint weights in {time.time() - t0:.3f}s.")

        if self.freeze_criteria is not None:
            t0 = g0_print("Freezing specified parameters...")
            freeze_parameters(self.model, self.freeze_criteria)
            g0_print(f"Froze specified parameters in {time.time() - t0:.3f}s.")

        g0_print("Watching model gradients with W&B...")
        cast(WandbLogger, self.trainer.logger).watch(self.model)

        g0_print("Done loading & configuring model.")

    def on_train_start(self):
        self.print(f"\nResetting model caches for training...\n")
        self.model.reset_caches()

    def compute_loss(self, inputs, targets):
        logits = self.model(input_ids=inputs)
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
        )
