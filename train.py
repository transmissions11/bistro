import torch

import lightning as L

from pathlib import Path

from lit_gpt.tokenizer import Tokenizer

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from utils.warnings import suppress_uncontrollable_warnings, elevate_important_warnings

from datetime import datetime

from lit_datamodule import LitDataModule
from lit_model import LitModel

from model import Config


devices = 1
micro_batch_size = 1
gradient_accumulation_iters = 1

epochs = 1

num_soft_prompt_tkns = 20
soft_prompt_tkn = "✅"

learning_rate = 3e-2
warmup_ratio = 0.05  # Spend 5% of training steps warming up.
weight_decay = 0.00  # Generally not used for finetuning.

val_split_ratio = 0.05  # 5% of training dataset.
val_batches = 1  # 100% of validation dataset.
tokens_to_sample = 8
val_check_interval = 0.0001  # After every 5% of training steps.

freeze_criteria = lambda name: "soft_prompt" not in name

project = "bistro"
run_name = datetime.now().strftime("%m-%d+%H:%M:%S")


hparams = {
    k: v
    for k, v in locals().items()
    if isinstance(v, (int, float, str)) and not k.startswith("_")
}


def main(data_dir: Path, checkpoint_dir: Path):
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
        dirpath=f"checkpoints/trained/{project}",
        filename="{epoch}-{step}-{val_loss:.2f}",
    )

    wandb_logger = WandbLogger(
        project=project,
        name=run_name,
        config=hparams,  # TODO: Ensure this includes parameters passed to main!
    )

    trainer = L.Trainer(
        devices=devices,
        strategy="auto",
        max_epochs=epochs,
        deterministic="warn",
        precision="bf16-true",
        logger=wandb_logger,
        limit_val_batches=val_batches,
        val_check_interval=val_check_interval,
        accumulate_grad_batches=gradient_accumulation_iters,
        num_sanity_val_steps=0,  # We run validate() before fit() already, so no need.
        callbacks=[LearningRateMonitor(logging_interval="step"), checkpoint_callback],
    )

    tokenizer = Tokenizer(checkpoint_dir)

    model = LitModel(
        model_config=Config.from_name(name=checkpoint_dir.name),
        tokenizer=tokenizer,
        checkpoint_path=checkpoint_dir / "lit_model.pth",
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        tokens_to_sample=tokens_to_sample,
        num_soft_prompt_tkns=num_soft_prompt_tkns,
        soft_prompt_tkn=soft_prompt_tkn,
        freeze_criteria=freeze_criteria,
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


def setup(
    data_dir: Path = Path("data"),
    checkpoint_dir: Path = Path("checkpoints/lmsys/vicuna-7b-v1.5"),
):
    main(data_dir, checkpoint_dir)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(setup)
