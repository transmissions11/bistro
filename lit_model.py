import time
import torch

import lightning as L

from pathlib import Path

from typing import Callable, Optional, cast

from lightning.pytorch.loggers import WandbLogger


from transformers import AutoModelForImageClassification


class LitModel(L.LightningModule):
    def __init__(
        self,
        ################################
        learning_rate: float,
        warmup_ratio: float,
        ################################
        weight_decay: float,
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

    def on_after_backward(self) -> None:
        super().on_after_backward()
        if self.current_epoch != 0:
            return

        # This function is useful for debuging the following error:
        # RuntimeError: It looks like your LightningModule has parameters that were not used in producing the loss returned by training_step.
        for name, p in self.named_parameters():
            if p.grad is None:
                print("unused parameter (check code or freeze it):", name)

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

        model_id = "google/siglip-so400m-patch14-384"
        self.model = AutoModelForImageClassification.from_pretrained(
            model_id,
            problem_type="multi_label_classification",
            id2label={0: "lturn", 1: "rturn", 2: "noturn"},
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
