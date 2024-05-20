import time
import torch

import lightning as L

from pathlib import Path

from functools import partial

from utils.lr_sched import cosine_with_linear_warmup

from typing import Callable, Optional, Tuple, cast

from lightning.pytorch.loggers import WandbLogger

from transformers import AutoModelForImageClassification, AutoConfig, logging


class LitModel(L.LightningModule):
    def __init__(
        self,
        model_id: str,
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
        pixel_values, labels = batch

        outputs = self.model(pixel_values=pixel_values, labels=labels)

        self.log("train_loss", outputs.loss)

        return outputs.loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        pixel_values, labels = batch

        outputs = self.model(pixel_values=pixel_values, labels=labels)

        self.log("val_loss", outputs.loss, on_epoch=True, prog_bar=True, sync_dist=True)

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

        config = AutoConfig.from_pretrained(
            self.hparams.model_id,
            problem_type="multi_label_classification",
            # TODO: Auto-derive mapping from the dataset?
            id2label={0: "lturn", 1: "rturn", 2: "noturn"},
        )

        # To avoid the error "It looks like your LightningModule has parameters
        # that were not used in producing the loss returned by training_step."
        # Option added in https://github.com/huggingface/transformers/pull/30814.
        config.vision_config.vision_use_head = False

        # To avoid "You should probably TRAIN this model on a down-stream task"
        # warning. See: https://github.com/huggingface/transformers/issues/5421
        if self._trainer and not self.trainer.is_global_zero:
            logging.set_verbosity(logging.ERROR)

        self.model = AutoModelForImageClassification.from_pretrained(
            config._name_or_path,
            config=config,
        )

        g0_print(f"Initialized model in {time.time() - t0:.3f}s.")

        if self.requires_grad is not None:
            t0 = g0_print("Toggling requires_grad on specified model parameters...")
            for name, param in self.model.named_parameters():
                param.requires_grad = self.requires_grad(name)
            g0_print(f"Toggled requires_grad on parameters in {time.time() - t0:.3f}s.")

        if self.watch_gradients:
            g0_print("Watching model gradients with W&B...")
            cast(WandbLogger, self.trainer.logger).watch(self.model)

        g0_print("Done loading & configuring model.")
