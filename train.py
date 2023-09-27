import torch

import lightning as L

from pathlib import Path

from typing import List, Optional

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
    micro_batch_size: int = 4,
    gradient_accumulation_iters: int = 1,
    #################################################################
    epochs: int = 1,
    #################################################################
    num_soft_prompt_tkns: int = 20,
    soft_prompt_tkn: str = "✅",
    #################################################################
    learning_rate: float = 6e-2,
    warmup_ratio: float = 0.05,  # Spend 5% of training steps warming.
    weight_decay: float = 0.00,  # Generally not used for finetuning.
    #################################################################
    val_split_ratio: float = 0.05,  # 5% of training dataset.
    val_check_interval: float = 0.05,  # After very 5% of training.
    #################################################################
    params_to_freeze: Optional[List[str]] = None,
    params_to_train: Optional[List[str]] = ["soft_prompt"],
    #################################################################
    run_name: str = datetime.now().strftime("%m-%d+%H:%M:%S"),
):
    assert not (
        params_to_freeze and params_to_train
    ), "Provide either params_to_freeze or params_to_train, not both."

    hparams = {
        k: v
        for k, v in locals().items()
        if isinstance(v, (int, float, str, list, type(None))) and not k.startswith("_")
    }

    print(hparams)  # TODO: remove

    # Filter out incorrect or "out of our control" warnings
    # and elevate important ones we want to treat as errors.
    suppress_uncontrollable_warnings()
    elevate_important_warnings()

    # Enables using the fast Tensor Cores on NVIDIA GPUs.
    torch.set_float32_matmul_precision("high")

    L.seed_everything(1337, workers=True)

    checkpoint_callback = ModelCheckpoint(
        verbose=True,
        save_top_k=5,
        monitor="val_loss",
        dirpath=f"checkpoints/trained/{project}/{run_name}",
        filename="{epoch}-{step}-{val_loss:.2f}",
    )

    wandb_logger = WandbLogger(
        project=project,
        name=run_name,
        config=hparams,
    )

    trainer = L.Trainer(
        devices=devices,
        strategy="auto",
        max_epochs=epochs,
        deterministic="warn",
        precision="bf16-true",
        logger=wandb_logger,
        val_check_interval=val_check_interval,
        accumulate_grad_batches=gradient_accumulation_iters,
        num_sanity_val_steps=0,  # We run validate() before fit() already, so no need.
        callbacks=[LearningRateMonitor(logging_interval="step"), checkpoint_callback],
    )

    tokenizer = Tokenizer(base_model_dir)

    model = LitModel(
        model_config=Config.from_name(name=base_model_dir.name),
        tokenizer=tokenizer,
        checkpoint_path=base_model_dir / "lit_model.pth",
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        num_soft_prompt_tkns=num_soft_prompt_tkns,
        soft_prompt_tkn=soft_prompt_tkn,
        requires_grad=(
            # If params_to_freeze is set, freeze all
            # params except those in params_to_freeze.
            (lambda param: param.name not in params_to_freeze)
            if params_to_freeze is not None
            # If params_to_train is set, only train those params.
            else (lambda param: param.name in params_to_train)
            if params_to_train is not None
            # Otherwise, train everything.
            else None
        ),
    )

    datamodule = LitDataModule(
        data_dir=str(data_dir),
        tokenizer=tokenizer,
        micro_batch_size=micro_batch_size,
        val_split_ratio=val_split_ratio,
        num_soft_prompt_tkns=num_soft_prompt_tkns,
        soft_prompt_tkn=soft_prompt_tkn,
    )

    trainer.validate(model, datamodule=datamodule)
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)
