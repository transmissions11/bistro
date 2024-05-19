import time
import torch

import lightning as L

from pathlib import Path

from functools import partial

from utils.lr_sched import cosine_with_linear_warmup

from typing import Callable, Optional, Tuple, cast

from lightning.pytorch.loggers import WandbLogger

from lit_gpt import Config, Tokenizer

from utils.loss import compute_loss
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
        ################################
        learning_rate: float,
        warmup_ratio: float,
        min_lr_ratio: float,
        ################################
        weight_decay: float,
        betas: Tuple[float, float],
        ################################
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
        loss = compute_loss(self.model, input_ids=inputs, target_ids=targets)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        inputs, targets = batch["inputs"], batch["targets"]
        loss = compute_loss(self.model, input_ids=inputs, target_ids=targets)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)

        # Log a few sample inferences from the validation set to W&B.
        if batch_idx == 0:
            tokenizer = self.hparams.tokenizer

            prompt_end_tkns = tokenizer.encode(
                VICUNA_END_OF_USER_PROMPT_SEQUENCE,
                device=self.device,
                # bos/eos=False because the "end tokens"
                # will be in the middle of the sequence.
                bos=False,
                eos=False,
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
            betas=self.hparams.betas,
        )

        if self.trainer.is_global_zero:
            print("Estimated total steps:", self.trainer.estimated_stepping_batches)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    partial(
                        cosine_with_linear_warmup,
                        warmup_steps=int(
                            self.trainer.estimated_stepping_batches
                            * self.hparams.warmup_ratio
                        ),
                        learning_rate=self.hparams.learning_rate,
                        min_lr_ratio=self.hparams.min_lr_ratio,
                        total_steps=self.trainer.estimated_stepping_batches,
                    ),
                ),
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
            # self._trainer will be None if we're loading a checkpoint
            # for inference. Can't check self.trainer directly since
            # it will throw an error if a trainer is not connected.
            if self._trainer is None or self.trainer.is_global_zero:
                print(msg)

            return time.time()  # For timing convenience.

        t0 = g0_print("Initializing model...")
        self.model = GPT(config=self.hparams.model_config)
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
