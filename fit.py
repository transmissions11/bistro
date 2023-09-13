import os
import time
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.loggers import WandbLogger
from lit_gpt.tokenizer import Tokenizer
from lit_gpt.utils import lazy_load, check_valid_checkpoint_dir
from lit_datamodule import LitDataModule
from utils.params import mark_only_soft_prompt_as_trainable
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint

from lit_model import LitModel

from model import GPT, Config

devices = 1
micro_batch_size = 1
gradient_accumulation_iters = 3

epochs = 1

num_soft_prompt_tkns = 20
soft_prompt_tkn = "✅"

learning_rate = 3e-2
min_lr_ratio = 0.00  # Anneal to 0.
warmup_ratio = 0.05  # Spend 5% of training steps warming up.
weight_decay = 0.01  # TODO: Should we be using this for finetuning?

val_batches = 10
tokens_to_sample = 8
val_check_interval = 0.05


hparams = {
    k: v
    for k, v in locals().items()
    if isinstance(v, (int, float, str)) and not k.startswith("_")
}


def main(data_dir: Path, checkpoint_dir: Path):
    check_valid_checkpoint_dir(checkpoint_dir)

    tokenizer = Tokenizer(checkpoint_dir)
    config = Config.from_name(name=checkpoint_dir.name)
    checkpoint_path = checkpoint_dir / "lit_model.pth"
    print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}...")

    L.seed_everything(1337, workers=True)

    wandb_logger = WandbLogger(
        project="bistro",
        config=hparams,
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=10,
        save_last=True,
        # every_n_train_steps=100,
        monitor="val/loss",
        mode="min",
        dirpath="bistro_checkpoints/",
        filename="ckpt-{epoch:02d}-{val_loss:.2f}",
    )

    trainer = L.Trainer(
        devices=devices,
        strategy="auto",
        max_epochs=epochs,
        deterministic=True,
        precision="bf16-true",
        logger=wandb_logger,
        log_every_n_steps=10,  # Doesn't apply in validation loop.
        limit_val_batches=val_batches,
        val_check_interval=val_check_interval,
        accumulate_grad_batches=gradient_accumulation_iters,
        callbacks=[LearningRateMonitor(logging_interval="step"), checkpoint_callback],
    )

    # Can set empty_init=True if can also set strict=True below.
    # Otherwise some parameters may not get initialized properly.
    with trainer.init_module(empty_init=False):
        gpt = GPT(
            config,
            soft_prompt_tkn=tokenizer.token_to_id(soft_prompt_tkn),
            num_soft_prompt_tkns=num_soft_prompt_tkns,
        )

    with lazy_load(checkpoint_path) as checkpoint:
        gpt.load_state_dict(checkpoint, strict=False)

    model = LitModel(
        gpt,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        min_lr_ratio=min_lr_ratio,
        weight_decay=weight_decay,
        tokens_to_sample=tokens_to_sample,
        tokenizer=tokenizer,
    )

    mark_only_soft_prompt_as_trainable(model)

    datamodule = LitDataModule(
        data_dir=str(data_dir),
        batch_size=micro_batch_size,
        tokenizer=tokenizer,
        num_soft_prompt_tkns=num_soft_prompt_tkns,
        soft_prompt_tkn=soft_prompt_tkn,
    )

    wandb_logger.watch(model)

    trainer.fit(model, datamodule=datamodule)


def setup(
    data_dir: Path = Path("data/bistro"),
    checkpoint_dir: Path = Path("checkpoints/lmsys/vicuna-7b-v1.5"),
):
    main(data_dir, checkpoint_dir)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(setup)
