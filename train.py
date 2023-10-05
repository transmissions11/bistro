import torch

import lightning as L

from pathlib import Path

from typing import List, Optional

from pprintjson import pprintjson

from lit_gpt.tokenizer import Tokenizer

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from utils.warnings import suppress_uncontrollable_warnings, elevate_important_warnings

from datetime import datetime

from lit_datamodule import LitDataModule
from lit_model import LitModel

from model import Config


def main(
    project: str = "bistro",
    #################################################################
    data_dir: Path = Path("data"),
    base_model_dir: Path = Path("checkpoints/lmsys/vicuna-7b-v1.5"),
    #################################################################
    devices: int = 4,
    strategy: str = "auto",
    micro_batch_size: int = 2,
    gradient_accumulation_iters: int = 16,
    precision: str = "bf16-true",
    #################################################################
    max_time: Optional[str] = None,  # Specify with DD:HH:MM:SS format.
    epochs: int = 1,  # Make this -1 to train forever / until max_time.
    #################################################################
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.05,  # Spend 5% of training steps warming.
    weight_decay: float = 0.00,  # Generally not used for finetuning.
    #################################################################
    val_split_ratio: float = 0.05,  # 5% of training dataset.
    val_check_interval: float = 0.05,  # After every 5% of training.
    skip_starting_validation: bool = False,  # Useful for debugging.
    #################################################################
    # Set params_to_freeze to freeze specific parameters, set params_to_train
    # to freeze everything except specific parameters, or set both to None to
    # train everything. They are mutually exclusive, at least one must be None.
    params_to_freeze: Optional[List[str]] = None,
    params_to_train: Optional[List[str]] = None,
    #################################################################
    log_every_n_steps: int = 50,
    disable_wandb: bool = False,  # Also useful for debugging.
    watch_gradients: bool = False,  # Very slow if training many params.
    profiler: Optional[str] = None,  # Either simple, advanced, or None.
    #################################################################
    save_checkpoints: bool = True,
    save_top_k_checkpoints: int = 5,
    #################################################################
    run_name: str = datetime.now().strftime("%m-%d+%H:%M:%S"),
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
        strategy=strategy,
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
                    dirpath=f"checkpoints/trained/{project}/{run_name}",
                    filename="{epoch}-{step}-{val_loss:.2f}",
                )
            ]
            if save_checkpoints
            else []
        ),
    )

    tokenizer = Tokenizer(base_model_dir)

    model = LitModel(
        model_config=Config.from_name(name=base_model_dir.name),
        tokenizer=tokenizer,
        checkpoint_path=base_model_dir / "lit_model.pth",
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        requires_grad=(
            # If params_to_freeze is set, freeze all
            # params except those in params_to_freeze.
            (lambda name: name not in params_to_freeze)
            if params_to_freeze is not None
            # If params_to_train is set, only train those.
            else (lambda name: name in params_to_train)
            if params_to_train is not None
            # Otherwise, train everything.
            else None
        ),
        watch_gradients=watch_gradients,
    )

    datamodule = LitDataModule(
        data_dir=str(data_dir),
        tokenizer=tokenizer,
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
