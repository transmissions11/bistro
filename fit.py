import torch

from pathlib import Path

import lightning as L

from lit_gpt.tokenizer import Tokenizer
from lit_datamodule import LitDataModule
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint

from utils.warnings import suppress_uncontrollable_warnings, elevate_important_warnings

from lit_model import LitModel

from model import Config


devices = 1
micro_batch_size = 3
gradient_accumulation_iters = 1

epochs = 1

num_soft_prompt_tkns = 20
soft_prompt_tkn = "✅"

learning_rate = 3e-2
warmup_ratio = 0.05  # Spend 5% of training steps warming up.
weight_decay = 0.00  # Generally not used for finetuning.

val_batches = 100
tokens_to_sample = 8
val_check_interval = 0.1  # After every 10% of training steps.

freeze_criteria = lambda name: "soft_prompt" not in name


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

    torch.set_float32_matmul_precision("high")

    L.seed_everything(1337, workers=True)

    checkpoint_callback = ModelCheckpoint(
        verbose=True,
        save_top_k=5,
        monitor="val_loss",
        dirpath="bistro_checkpoints/",
        filename="{epoch}-{step}-{val_loss:.2f}",
    )

    trainer = L.Trainer(
        devices=devices,
        strategy="auto",
        max_epochs=epochs,
        deterministic="warn",
        precision="bf16-true",
        logger=WandbLogger(
            project="bistro",
            config=hparams,  # TODO: Ensure this includes parameters passed to main!
        ),
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
        batch_size=micro_batch_size,
        tokenizer=tokenizer,
        num_soft_prompt_tkns=num_soft_prompt_tkns,
        soft_prompt_tkn=soft_prompt_tkn,
    )

    trainer.validate(model, datamodule=datamodule)
    trainer.fit(model, datamodule=datamodule)


def setup(
    data_dir: Path = Path("data/bistro"),
    checkpoint_dir: Path = Path("checkpoints/lmsys/vicuna-7b-v1.5"),
):
    main(data_dir, checkpoint_dir)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(setup)
