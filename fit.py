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
from lightning.pytorch.callbacks import DeviceStatsMonitor

from lit_model import LitModel

from model import GPT, Config

devices = 1
micro_batch_size = 1
gradient_accumulation_iters = 3

# TODO: Make these hyperparameters?
num_soft_prompt_tkns = 20
soft_prompt_tkn = "✅"  # TODO: Make this work across multiple tokenizers.

learning_rate = 3e-2
min_learning_rate = 0
warmup_steps = 2000
weight_decay = 0.02


# Should we use https://github.com/omry/omegaconf?


def main(data_dir: Path, checkpoint_dir: Path, out_dir: Path):
    check_valid_checkpoint_dir(checkpoint_dir)

    tokenizer = Tokenizer(checkpoint_dir)
    config = Config.from_name(name=checkpoint_dir.name)
    checkpoint_path = checkpoint_dir / "lit_model.pth"
    print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}...")

    L.seed_everything(1337, workers=True)  # TODO: Do we need this?

    # TODO: I really just want to be able to attach all trainer params.
    # TODO: We could just have a config dict that gets passed into main
    # with everything, and use **config syntax to unpack it into trainer.
    # TODO: Then we'd just log that with log_hyperparams.
    # TODO: !!!!!! IF WE DO THIS, we should pass logger=False to save_hyperparams !!!
    wandb_logger = WandbLogger(
        project="bistro",
        config={
            "devices": devices,
            "micro_batch_size": micro_batch_size,
            "gradient_accumulation_iters": gradient_accumulation_iters,
        },
    )

    trainer = L.Trainer(
        devices=devices,
        strategy="auto",  # deepspeed/ddp
        precision="bf16-true",
        logger=wandb_logger,
        accumulate_grad_batches=gradient_accumulation_iters,
        max_epochs=1,
        log_every_n_steps=1,
        deterministic=True,  # TODO: Do we need this? Should we be using "warn"?
        callbacks=[LearningRateMonitor(logging_interval="step"), DeviceStatsMonitor()],
        # profiler="simple",
        # max_steps=1000,
    )

    # TODO: Try logging grads to wandb
    # https://pytorch-lightning.readthedocs.io/en/1.4.9/extensions/generated/pytorch_lightning.loggers.WandbLogger.html

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
        min_learning_rate=min_learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
    )

    mark_only_soft_prompt_as_trainable(model)

    datamodule = LitDataModule(
        data_dir=str(data_dir),
        # TODO: Should this be batch_size or micro_batch_size?
        batch_size=micro_batch_size,
        tokenizer=tokenizer,
        num_soft_prompt_tkns=num_soft_prompt_tkns,
        soft_prompt_tkn=soft_prompt_tkn,
    )

    trainer.fit(model, datamodule=datamodule)

    trainer.save_checkpoint(out_dir / "model_finetuned.pth")


def setup(
    data_dir: Path = Path("data/bistro"),
    checkpoint_dir: Path = Path("checkpoints/lmsys/vicuna-7b-v1.5"),
    out_dir: Path = Path("out/full/bistro"),
):
    main(data_dir, checkpoint_dir, out_dir)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(setup)
