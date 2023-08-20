import os
import time
from pathlib import Path

import lightning as L
import torch
from datasets import load_dataset, DatasetDict, Dataset
from lightning.fabric.strategies import FSDPStrategy
from lightning.pytorch.loggers import WandbLogger
from lit_gpt.speed_monitor import (
    SpeedMonitorFabric as SpeedMonitor,
)
from lit_gpt.tokenizer import Tokenizer
from lit_gpt.utils import (
    lazy_load,
    check_valid_checkpoint_dir,
    chunked_cross_entropy,
)
from tqdm import tqdm

from model import GPT, Config, Block
from sample import sample_model
from utils.batch import get_batch
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
learning_rate = 1
batch_size = 64 / devices  # TODO: Configure this better.
micro_batch_size = 1  # TODO: Set a larger value for this.
gradient_accumulation_iters = 3  # batch_size // micro_batch_size
assert gradient_accumulation_iters > 0

num_soft_prompt_tkns = 20
soft_prompt_tkn = "✅"  # TODO: Make this work across multiple tokenizers.

epoch_size = 10_000_000  # TODO: Set this based on the actual dataset dynamically.
num_epochs = 4

max_iters = num_epochs * (epoch_size // micro_batch_size) // devices
weight_decay = 0.02  # TODO: Should we be using this for finetuning?
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
    model: GPT,
    optimizer: torch.optim.Optimizer,
    datasets: DatasetDict,
    tokenizer: Tokenizer,
    out_dir: Path,
    speed_monitor: SpeedMonitor,
) -> None:
    fabric.print(
        f"starting val loss: {validate(fabric, model, datasets['validation'], tokenizer):.4f}"
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

    for iter_num in tqdm(range(max_iters)):
        # Linear warmup stage.
        if step_count <= warmup_steps:
            lr = learning_rate * step_count / warmup_steps
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        iter_t0 = time.time()

        input_ids, targets = get_batch(
            fabric,
            datasets["train"],
            tokenizer,
            micro_batch_size,
        )

        is_accumulating = (iter_num + 1) % gradient_accumulation_iters != 0

        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids)
            loss = chunked_cross_entropy(logits, targets, chunk_size=0)

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
            val_loss = validate(
                fabric,
                model,
                datasets["validation"],
                tokenizer,
            )
            t1 = time.time() - t0
            speed_monitor.eval_end(t1)
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
    model: GPT,
    val_dataset: Dataset,
    tokenizer: Tokenizer,
) -> torch.Tensor:
    fabric.print("Validating ...")

    model.eval()

    losses = torch.zeros(eval_iters)

    for k in range(eval_iters):
        input_ids, targets = get_batch(fabric, val_dataset, tokenizer, micro_batch_size)

        logits = model(input_ids)
        print(logits, targets)
        loss = chunked_cross_entropy(logits, targets, chunk_size=0)
        losses[k] = loss.item()

        # Target generating 5 examples.
        if k % (eval_iters // 5) == 0:
            tokens_out = 10

            sample = strip_right_pad(input_ids[0])
            target = strip_right_pad(targets[0])

            prompt_end_idx = find_subtensor_end(
                sample, VICUNA_END_OF_USER_PROMPT_SEQUENCE
            )

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

    datasets = load_dataset("parquet", data_dir=f"{data_dir}")

    config = Config.from_name(name=checkpoint_dir.name)
    checkpoint_path = checkpoint_dir / "lit_model.pth"
    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}...")
    with fabric.init_module(empty_init=False):
        print(tokenizer.token_to_id(soft_prompt_tkn))
        model = GPT(
            config,
            soft_prompt_tkn=tokenizer.token_to_id(soft_prompt_tkn),
            num_soft_prompt_tkns=num_soft_prompt_tkns,
        )
    with lazy_load(checkpoint_path) as checkpoint:
        model.load_state_dict(checkpoint, strict=False)

    #################################################################

    print(datasets["train"][0])

    # Add soft prompt to the beginning of each input's prompt.
    # TODO: Should we just tokenize and Vicuna format here vs later?
    # TODO: Why does this map over the 20,000,000 inputs twice (each split only has 10m)?
    datasets = datasets.map(
        lambda x: {
            "prompt": f"{soft_prompt_tkn * num_soft_prompt_tkns} {x['prompt']}",
            "response": x["response"],
        },
        num_proc=8,
    )

    print(datasets["train"][0])

    mark_only_soft_prompt_as_trainable(model)

    #################################################################

    param_breakdown = get_param_breakdown(model)
    fabric.print(
        f"Number of trainable and non-trainable parameters: "
        f"{param_breakdown['num_trainable_params'],} | {param_breakdown['num_non_trainable_params'],}"
    )
    fabric.print(f"Trainable parameters: {param_breakdown['trainable_param_names']}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    model, optimizer = fabric.setup(model, optimizer)

    fabric.seed_everything(1337 + fabric.global_rank)

    train(
        fabric,
        model,
        optimizer,
        datasets,
        tokenizer,
        out_dir,
        speed_monitor,
    )

    save_checkpoint(fabric, model, out_dir / "bistro_model_finetuned.pth")


def setup(
    data_dir: Path = Path("data/chess"),
    checkpoint_dir: Path = Path("checkpoints/lmsys/vicuna-7b-v1.5"),
    out_dir: Path = Path("out/full/chess"),
    # TODO: Try precision="transformer-engine" (https://github.com/Lightning-AI/lightning/pull/17597)
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
