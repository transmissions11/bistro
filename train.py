import torch

import lightning as L

from pathlib import Path

from typing import List, Optional

from pprintjson import pprintjson


from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.strategies import DDPStrategy

from utils.debugging import iexd
from utils.naming import get_clean_commit_msg, get_safe_ckpt_dirpath
from utils.warnings import suppress_uncontrollable_warnings, elevate_important_warnings

from lit_datamodule import LitDataModule
from lit_model import LitModel

# TODO: whu unsued params


@iexd  # Will drop into ipdb if an exception is raised on rank zero.
def main(
    project: str = "siglip-classifier",
    ####################################################################
    data_dir: Path = Path("data"),
    ####################################################################
    devices: int = -1,  # -1 for all available GPUs, 1 for 1 GPU, etc.
    strategy: str = "auto",
    micro_batch_size: int = 32,
    gradient_accumulation_iters: int = 4,
    precision: str = "bf16-true",
    ####################################################################
    max_time: Optional[str] = None,  # Specify with DD:HH:MM:SS format.
    epochs: int = 1,  # Make this -1 to train forever / until max_time.
    ####################################################################
    learning_rate: float = 5e-5,
    warmup_ratio: float = 0.05,  # Spend 5% of training steps warming.
    weight_decay: float = 0.00,  # Generally not used for finetuning.
    ####################################################################
    val_split_ratio: float = 0.05,  # 5% of training dataset.
    val_check_interval: float = 0.05,  # After every 5% of training.
    skip_starting_validation: bool = False,  # Useful for debugging.
    ####################################################################
    # Set params_to_freeze to freeze specific parameters, set params_to_train
    # to freeze everything except specific parameters, or set both to None to
    # train everything. They are mutually exclusive, at least one must be None.
    params_to_freeze: Optional[List[str]] = None,
    params_to_train: Optional[List[str]] = None,
    ####################################################################
    log_every_n_steps: int = 1,
    disable_wandb: bool = False,  # Also useful for debugging.
    watch_gradients: bool = False,  # Very slow if training many params.
    profiler: Optional[str] = None,  # Either simple, advanced, or None.
    ####################################################################
    save_checkpoints: bool = True,
    save_top_k_checkpoints: int = 5,
    ####################################################################
    run_name: str = get_clean_commit_msg(),  # Used for ckpt dirpath and W&B.
):
    """
    Bistro: ♪ The finest of the finer things, 24 hours a day, 7 days a week ♪
    """

    assert not (
        params_to_freeze and params_to_train
    ), "Provide either params_to_freeze or params_to_train, not both."

    hparams = {
        k: v
        for k, v in locals().items()
        if isinstance(v, (int, float, str, list, type(None))) and not k.startswith("_")
    }

    # Filter out incorrect or "out of our control" warnings
    # and elevate important ones we want to treat as errors.
    suppress_uncontrollable_warnings()
    elevate_important_warnings()

    # Enables using the fast Tensor Cores on NVIDIA GPUs.
    torch.set_float32_matmul_precision("high")

    L.seed_everything(1337, workers=True)

    trainer = L.Trainer(
        devices=devices,
        strategy=DDPStrategy(find_unused_parameters=True),
        max_epochs=epochs,
        max_time=max_time,
        profiler=profiler,
        deterministic="warn",
        precision=precision,
        val_check_interval=val_check_interval,
        log_every_n_steps=log_every_n_steps,
        enable_checkpointing=save_checkpoints,
        accumulate_grad_batches=gradient_accumulation_iters,
        num_sanity_val_steps=0,  # We run validate() before fit() already, so no need.
        logger=(
            WandbLogger(
                project=project,
                name=run_name,
                config=hparams,
            )
            if not disable_wandb
            else False
        ),
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
        ]
        + (
            [
                ModelCheckpoint(
                    verbose=True,
                    monitor="val_loss",
                    save_top_k=save_top_k_checkpoints,
                    dirpath=get_safe_ckpt_dirpath(project, run_name),
                    filename="{epoch}-{step}-{val_loss:.2f}",
                )
            ]
            if save_checkpoints
            else []
        ),
    )

    model = LitModel(
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        requires_grad=(
            # If params_to_freeze is set, freeze all
            # params except those in params_to_freeze.
            (lambda name: name not in params_to_freeze)
            if params_to_freeze is not None
            # If params_to_train is set, only train those.
            else (
                (lambda name: name in params_to_train)
                if params_to_train is not None
                # Otherwise, train everything.
                else None
            )
        ),
        watch_gradients=watch_gradients,
    )

    datamodule = LitDataModule(
        data_dir=str(data_dir),
        micro_batch_size=micro_batch_size,
        val_split_ratio=val_split_ratio,
    )

    if trainer.is_global_zero:
        print("Training with the following hyperparameters:")
        pprintjson(hparams)

    if not skip_starting_validation:
        trainer.validate(model, datamodule=datamodule)

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)
