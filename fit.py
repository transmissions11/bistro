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

from lit_model import LitModel

from model import GPT, Config

devices = 1

# Hyperparameters.
learning_rate = 1  # TODO: This is duplicated in lit_model!
batch_size = 64 / devices  # TODO: Configure this better.
micro_batch_size = 1  # TODO: Set a larger value for this.
gradient_accumulation_iters = 3  # batch_size // micro_batch_size
assert gradient_accumulation_iters > 0

num_soft_prompt_tkns = 20
soft_prompt_tkn = "✅"  # TODO: Make this work across multiple tokenizers.

# TODO: ALl of this logic is fucked. We currently do (1 * num_devices) epochs I think.
# We should fix this at some point !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
epoch_size = 10_000_000
num_epochs = 4
max_iters = num_epochs * (epoch_size // micro_batch_size) // devices

warmup_steps = (
    2 * (epoch_size // micro_batch_size) // devices // gradient_accumulation_iters
)  # 2 epochs — TODO: Set this to some industry standard (5%?)

hparams = {
    k: v
    for k, v in locals().items()
    if isinstance(v, (int, float, str)) and not k.startswith("_")
}


def main(data_dir: Path, checkpoint_dir: Path, out_dir: Path):
    print("Hyperparams:", hparams)

    check_valid_checkpoint_dir(checkpoint_dir)

    tokenizer = Tokenizer(checkpoint_dir)
    config = Config.from_name(name=checkpoint_dir.name)
    checkpoint_path = checkpoint_dir / "lit_model.pth"
    print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}...")

    L.seed_everything(1337, workers=True)

    trainer = L.Trainer(
        devices=devices,
        # strategy="deepspeed",
        precision="bf16-true",
        logger=WandbLogger(project="bistro"),
        accumulate_grad_batches=gradient_accumulation_iters,
        max_epochs=1,
        log_every_n_steps=1,
        deterministic=True,
    )

    # TODO: Should empty_init be True or False?
    with trainer.init_module(empty_init=False):
        gpt = GPT(
            config,
            soft_prompt_tkn=tokenizer.token_to_id(soft_prompt_tkn),
            num_soft_prompt_tkns=num_soft_prompt_tkns,
        )
    with lazy_load(checkpoint_path) as checkpoint:
        gpt.load_state_dict(checkpoint, strict=False)

    model = LitModel(gpt)
    mark_only_soft_prompt_as_trainable(model)

    # fabric.seed_everything(1337 + fabric.global_rank)

    datamodule = LitDataModule(
        data_dir=data_dir,
        # TODO: Should this be batch_size or micro_batch_size?
        batch_size=micro_batch_size,
        tokenizer=tokenizer,
        num_soft_prompt_tkns=num_soft_prompt_tkns,
        soft_prompt_tkn=soft_prompt_tkn,
    )

    trainer.fit(model, datamodule=datamodule)

    trainer.save_checkpoint(out_dir / "bistro_model_finetuned.pth")


def setup(
    data_dir: Path = Path("data"),
    checkpoint_dir: Path = Path("checkpoints/lmsys/vicuna-7b-v1.5"),
    out_dir: Path = Path("out/full/bistro"),
):
    main(data_dir, checkpoint_dir, out_dir)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(setup)
