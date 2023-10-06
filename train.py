import torch

import lightning as L

from pathlib import Path

from typing import Optional

from pprintjson import pprintjson

from lit_gpt.tokenizer import Tokenizer

from lightning.pytorch.loggers import WandbLogger

from utils.warnings import suppress_uncontrollable_warnings, elevate_important_warnings

from datetime import datetime

from lit_datamodule import LitDataModule
from lit_model import LitModel

from model import Config


def main(
    project: str = "hard-prompting",
    #################################################################
    data_dir: Path = Path("data"),
    base_model_dir: Path = Path("checkpoints/lmsys/vicuna-7b-v1.5"),
    #################################################################
    devices: int = 1,
    strategy: str = "auto",
    precision: str = "bf16-true",
    #################################################################
    max_time: Optional[str] = None,  # Specify with DD:HH:MM:SS format.
    epochs: int = -1,  # Make this -1 to train forever / until max_time.
    #################################################################
    num_hard_prompt_tkns: int = 20,
    hard_prompt_tkn: str = "✅",
    #################################################################
    val_split_ratio: float = 0.01,  # 1% of training dataset.
    val_check_interval: float = 0.01,  # After every 1% of training.
    skip_starting_validation: bool = False,  # Useful for debugging.
    #################################################################
    log_every_n_steps: int = 10,
    disable_wandb: bool = False,  # Also useful for debugging.
    profiler: Optional[str] = None,  # Either simple, advanced, or None.
    #################################################################
    run_name: str = datetime.now().strftime("%m-%d+%H:%M:%S"),
):
    """
    Bistro: ♪ The finest of the finer things, 24 hours a day, 7 days a week ♪
    """

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
        #########################
        limit_train_batches=1,
        limit_val_batches=0,
        ##########################
        val_check_interval=val_check_interval,
        log_every_n_steps=log_every_n_steps,
        enable_checkpointing=False,
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
    )

    tokenizer = Tokenizer(base_model_dir)

    model = LitModel(
        model_config=Config.from_name(name=base_model_dir.name),
        tokenizer=tokenizer,
        hard_prompt_tkn=tokenizer.token_to_id(hard_prompt_tkn),
        num_hard_prompt_tkns=num_hard_prompt_tkns,
        checkpoint_path=base_model_dir / "lit_model.pth",
    )

    datamodule = LitDataModule(
        data_dir=str(data_dir),
        tokenizer=tokenizer,
        val_split_ratio=val_split_ratio,
        num_hard_prompt_tkns=num_hard_prompt_tkns,
        hard_prompt_tkn=hard_prompt_tkn,
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
