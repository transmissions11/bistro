import time
import torch

from torch.nn import functional as F

import lightning as L

from pathlib import Path

from typing import Callable, Optional, cast

from lightning.pytorch.loggers import WandbLogger

from lit_gpt import Config, Tokenizer

from utils.inference import inference_model
from utils.tensors import find_subtensor_end
from utils.padding import strip_right_pad, ignored_tkn
from utils.vicuna import VICUNA_END_OF_USER_PROMPT_SEQUENCE

from model import GPT


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
        num_soft_prompt_tkns: int,
        soft_prompt_tkn: str,
        ###############################
        # If None, will use random weights.
        checkpoint_path: Optional[Path] = None,
        # If None, all parameters will be trained.
        requires_grad: Optional[Callable[[str], bool]] = None,
        # If True, will watch & log gradients to W&B.
        # Will grind to a halt if training many params.
        watch_gradients: bool = False,
    ):
        super().__init__()

        self.model = None  # This will get set in configure_model.

        # Assign these manually as they don't pickle well
        # or shouldn't be saved via save_hyperparameters.
        self.requires_grad = requires_grad
        self.checkpoint_path = checkpoint_path
        self.watch_gradients = watch_gradients

        # logger=False since we already log hparams manually in train.py.
        self.save_hyperparameters(
            ignore=["checkpoint_path", "requires_grad", "watch_gradients"], logger=False
        )

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
            loss.to(torch.float32),
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        # Log a few sample inferences from the validation set to W&B.
        if batch_idx == 0:
            tokenizer = self.hparams.tokenizer

            prompt_end_tkns = tokenizer.encode(
                VICUNA_END_OF_USER_PROMPT_SEQUENCE, device=self.device
            )

            def process_val_sample(sample, target):
                sample = strip_right_pad(sample)
                target = strip_right_pad(target)

                input_ids = sample[: find_subtensor_end(sample, prompt_end_tkns) + 1]

                return (
                    tokenizer.decode(input_ids),
                    tokenizer.decode(
                        inference_model(
                            self.model,
                            input_ids,
                            temperature=0.00,  # Sample 100% greedily.
                            max_new_tokens=100,  # Should hit an eos token first.
                            eos_id=tokenizer.eos_id,
                        )
                    ),
                    # Note: target != ignored_tkn strips away ignored_tkn tokens entirely,
                    # which may lead to confusion if ignored_tkn is used between real tokens.
                    tokenizer.decode(target[target != ignored_tkn]),
                )

            self.logger.log_text(
                key="val_samples",
                columns=["input", "output", "target"],
                data=[
                    process_val_sample(inputs[i], targets[i])
                    for i in range(len(inputs))  # Full batch.
                ],
            )

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
        # Ensure this function is idempotent, as
        # the trainer may call it multiple times.
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

        if self.checkpoint_path is not None:
            t0 = g0_print(f"Loading checkpoint weights from {self.checkpoint_path}...")
            self.model.load_state_dict(
                # mmap=True requires checkpoint_path to be a str.
                torch.load(str(self.checkpoint_path), mmap=True),
                strict=False,
                assign=True,
            )
            g0_print(f"Loaded checkpoint weights in {time.time() - t0:.3f}s.")

        if self.requires_grad is not None:
            t0 = g0_print("Toggling requires_grad on specified model parameters...")
            for name, param in self.model.named_parameters():
                param.requires_grad = self.requires_grad(name)
            g0_print(f"Toggled requires_grad on parameters in {time.time() - t0:.3f}s.")

        if self.watch_gradients:
            g0_print("Watching model gradients with W&B...")
            cast(WandbLogger, self.trainer.logger).watch(self.model)

        g0_print("Done loading & configuring model.")

    def on_train_start(self):
        self.print(f"\nResetting model caches for training...\n")
        self.model.reset_caches()

    def compute_loss(self, inputs, targets):
        logits = self.model(input_ids=inputs)
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=ignored_tkn
        )
