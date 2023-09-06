import os
import time
from pathlib import Path

import lightning as L
import torch
from datasets import load_dataset
from lightning.fabric.strategies import FSDPStrategy
from lightning.pytorch.loggers import WandbLogger
from lit_gpt.speed_monitor import (
    SpeedMonitorFabric as SpeedMonitor,
)
from lit_gpt.tokenizer import Tokenizer
from lit_gpt.utils import lazy_load, check_valid_checkpoint_dir
from lit_datamodule import LitDataModule

from lit_model import LitModel

from model import GPT, Config, Block
from sample import sample_model
from utils.padding import strip_right_pad
from utils.params import get_param_breakdown, mark_only_soft_prompt_as_trainable
from utils.save import save_checkpoint
from utils.tensors import find_subtensor_end
from utils.vicuna import VICUNA_END_OF_USER_PROMPT_SEQUENCE

log_interval = 1
eval_interval, eval_iters = 50, 100
save_interval = 9999999999  # 600

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


def train(
    fabric: L.Fabric,
    model: LitModel,
    optimizer: torch.optim.Optimizer,
    datamodule: LitDataModule,
    tokenizer: Tokenizer,
    out_dir: Path,
    speed_monitor: SpeedMonitor,
) -> None:
    fabric.print(
        f"starting val loss: {validate(fabric, model, datamodule.val_dataloader(), tokenizer):.4f}"
    )

    measured_flops = 0.0  # TODO: Should get this working again.
    # with torch.device("meta"):
    #     meta_model = GPT(model.config, num_tokens_in_soft_prompt)
    #     x = torch.randint(0, 1, (micro_batch_size, model.config.block_size))
    #     measured_flops = measure_flops(meta_model, x)
    #     fabric.print(
    #         f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}"
    #     )
    #     del meta_model, x

    step_count = 0
    total_lengths = 0
    total_t0 = time.time()

    for iter_num, batch in enumerate(datamodule.train_dataloader()):
        # Linear warmup stage.
        if step_count <= warmup_steps:
            lr = learning_rate * step_count / warmup_steps
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        iter_t0 = time.time()

        is_accumulating = (iter_num + 1) % gradient_accumulation_iters != 0

        with fabric.no_backward_sync(model, enabled=is_accumulating):
            # TODO: Ideally moving to device gets done for us automatically!
            loss = model.training_step(batch.to(fabric.device), iter_num)

            fabric.backward(loss / gradient_accumulation_iters)

        if not is_accumulating:
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1

        t1 = time.time()
        total_lengths += input_ids.size(1)
        speed_monitor.on_train_batch_end(
            (iter_num + 1) * micro_batch_size,
            t1 - total_t0,
            # Assumes that device FLOPs are the same
            # and all devices have the same batch size.
            fabric.world_size,
            flops_per_batch=measured_flops,
            lengths=total_lengths,
        )
        if iter_num % log_interval == 0:
            fabric.log("train/loss", loss.item())
            fabric.print(
                f"iter {iter_num} step {step_count}: loss {loss.item():.4f}, iter time:"
                f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
            )

        if not is_accumulating and step_count % eval_interval == 0:
            t0 = time.time()
            val_loss = validate(fabric, model, datamodule.val_dataloader(), tokenizer)
            t1 = time.time() - t0
            speed_monitor.eval_end(t1)
            # TODO: W&B table to show examples.
            fabric.log("val/loss", val_loss)
            fabric.print(
                f"step {iter_num}: val loss {val_loss:.4f}, val time: {t1 * 1000:.2f}ms"
            )
            fabric.barrier()

        if not is_accumulating and step_count % save_interval == 0:
            save_checkpoint(fabric, model, out_dir / f"iter-{iter_num:06d}-ckpt.pth")


@torch.no_grad()
def validate(
    fabric: L.Fabric,
    model: LitModel,
    val_dataloader: torch.utils.data.DataLoader,
    tokenizer: Tokenizer,
) -> torch.Tensor:
    fabric.print("Validating ...")

    model.eval()

    losses = torch.zeros(eval_iters)

    for k, batch in enumerate(val_dataloader):
        # TODO: Ideally moving to device gets done for us automatically!
        loss = model.validation_step(batch.to(fabric.device), k)
        losses[k] = loss.item()

        # Target generating 5 examples.
        if k % (eval_iters // 5) == 0:
            tokens_out = 10

            sample = strip_right_pad(batch[0]["input_ids"])
            target = strip_right_pad(batch[0]["targets"])

            prompt_end_idx = find_subtensor_end(
                sample,
                tokenizer.encode(
                    # TODO: Why did I have to do device=fabric.device here, and not in other places?
                    VICUNA_END_OF_USER_PROMPT_SEQUENCE,
                    device=fabric.device,
                ),
            )

            # TODO: Print and compare to a baseline w/o the soft prompt.

            print(f"Input: {tokenizer.decode(sample[:prompt_end_idx + 1])}")
            output = sample_model(
                model,
                idx=sample[: prompt_end_idx + 1],
                max_new_tokens=10,
                temperature=0.01,
            )[-tokens_out:]
            print(f"Output:", tokenizer.decode(output))
            target[target == -1] = 0  # TODO: Just show the relevant targets.
            print(f"Target:", tokenizer.decode(target))
            print("\n\n")

    model.train()

    return losses.mean().item()


def main(fabric: L.Fabric, data_dir: Path, checkpoint_dir: Path, out_dir: Path):
    fabric.print("Hyperparams:", hparams)

    check_valid_checkpoint_dir(checkpoint_dir)

    speed_monitor = SpeedMonitor(fabric, window_size=50, time_unit="seconds")

    fabric.seed_everything(1337)  # Same seed for every process to init model (FSDP)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    tokenizer = Tokenizer(checkpoint_dir)

    config = Config.from_name(name=checkpoint_dir.name)
    checkpoint_path = checkpoint_dir / "lit_model.pth"
    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}...")
    with fabric.init_module(empty_init=False):
        gpt = GPT(
            config,
            soft_prompt_tkn=tokenizer.token_to_id(soft_prompt_tkn),
            num_soft_prompt_tkns=num_soft_prompt_tkns,
        )
    with lazy_load(checkpoint_path) as checkpoint:
        gpt.load_state_dict(checkpoint, strict=False)

    model = LitModel(gpt)

    #################################################################

    mark_only_soft_prompt_as_trainable(model)

    #################################################################

    param_breakdown = get_param_breakdown(model)
    fabric.print(
        f"Number of trainable and non-trainable parameters: "
        f"{param_breakdown['num_trainable_params']} | {param_breakdown['num_non_trainable_params']}"
    )
    fabric.print(f"Trainable parameters: {param_breakdown['trainable_param_names']}")

    optimizer = model.configure_optimizers()

    model, optimizer = fabric.setup(model, optimizer)

    ################################################################

    datamodule = LitDataModule(
        data_dir=data_dir,
        # TODO: Should this be batch_size or micro_batch_size?
        batch_size=micro_batch_size,
        tokenizer=tokenizer,
        num_soft_prompt_tkns=num_soft_prompt_tkns,
        soft_prompt_tkn=soft_prompt_tkn,
    )
    datamodule.prepare_data()
    datamodule.setup("fit")

    ################################################################

    fabric.seed_everything(1337 + fabric.global_rank)

    train(
        fabric,
        model,
        optimizer,
        datamodule,
        tokenizer,
        out_dir,
        speed_monitor,
    )

    save_checkpoint(fabric, model, out_dir / "bistro_model_finetuned.pth")


def setup(
    data_dir: Path = Path("data"),
    checkpoint_dir: Path = Path("checkpoints/lmsys/vicuna-7b-v1.5"),
    out_dir: Path = Path("out/full/bistro"),
    # TODO: Try "transformer-engine" (https://github.com/Lightning-AI/lightning/pull/17597)
    # TODO: Make this a W&B sweep param (bf16-true, bf16-mixed, 16-true, 16-mixed, fp8, 64, 32)
    precision: str = "bf16-true",
):
    if devices > 1:
        strategy = FSDPStrategy(
            auto_wrap_policy={Block},
            activation_checkpointing_policy={Block},
            state_dict_type="full",
            limit_all_gathers=True,
            cpu_offload=False,
        )
    else:
        strategy = "auto"

    fabric = L.Fabric(
        devices=devices,
        strategy=strategy,
        precision=precision,
        loggers=WandbLogger(project="bistro"),
    )

    fabric.launch(main, data_dir, checkpoint_dir, out_dir)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(setup)
