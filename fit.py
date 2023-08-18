import os
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import lightning as L

from datasets import load_dataset, DatasetDict, Dataset
from lightning.fabric.strategies import FSDPStrategy
from lightning.pytorch.loggers import WandbLogger

from lit_gpt.tokenizer import Tokenizer
from lit_gpt.utils import (
    lazy_load,
    check_valid_checkpoint_dir,
    chunked_cross_entropy,
)
from lit_gpt.speed_monitor import (
    SpeedMonitorFabric as SpeedMonitor,
    measure_flops,
)
from tqdm import tqdm

from sample import sample_model
from model import GPT, Config, Block
from utils.padding import ignored_tkn, pad_tkn, pad_right, strip_right_pad

log_interval = 1
eval_interval, eval_iters = 50, 100
save_interval = 9999999999  # 600

devices = 1

# change this value to force a maximum sequence length
override_max_seq_length = None

# TODO: BETTER WAY TO DO THIS
num_tokens_in_soft_prompt = 20

# Hyperparameters
learning_rate = 1
batch_size = 64 / devices
micro_batch_size = 1  # TODO: wtf is going on here
gradient_accumulation_iters = 3  # batch_size // micro_batch_size
assert gradient_accumulation_iters > 0

epoch_size = 50000  # train dataset size
num_epochs = 5

max_iters = num_epochs * (epoch_size // micro_batch_size) // devices
weight_decay = 0.02  # TODO maybe dont use this lol
warmup_steps = (
    2 * (epoch_size // micro_batch_size) // devices // gradient_accumulation_iters
)  # 2 epochs

hparams = {
    k: v
    for k, v in locals().items()
    if isinstance(v, (int, float, str)) and not k.startswith("_")
}


def mark_only_soft_prompt_as_trainable(model: GPT) -> None:
    """Sets `requires_grad=False` for all non-soft-prompt weights."""
    for name, param in model.named_parameters():
        param.requires_grad = "soft_prompt" in name


def format_prompt(problem: str, resp: str) -> str:
    system_prompt = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions. "
        "USER: {soft_prompt_tkns} {problem} = ASSISTANT: {resp}"
    )
    return system_prompt.format(
        problem=problem,
        resp=resp,
        soft_prompt_tkns=(
            "✅" * num_tokens_in_soft_prompt
        ),  # TODO: figure out how to add custom tkns
    )


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


def main(fabric: L.Fabric, data_dir: Path, checkpoint_dir: Path, out_dir: Path):
    fabric.print(hparams)
    check_valid_checkpoint_dir(checkpoint_dir)

    speed_monitor = SpeedMonitor(fabric, window_size=50, time_unit="seconds")

    fabric.seed_everything(1337)  # same seed for every process to init model (FSDP)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    dataset = load_dataset("parquet", data_dir=f"{data_dir}")

    config = Config.from_name(name=checkpoint_dir.name)
    checkpoint_path = checkpoint_dir / "lit_model.pth"
    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}")
    with fabric.init_module(empty_init=False):
        model = GPT(config, num_tokens_in_soft_prompt)
    with lazy_load(checkpoint_path) as checkpoint:
        model.load_state_dict(checkpoint, strict=False)

    mark_only_soft_prompt_as_trainable(model)

    # todo: gigacursed code
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    num_params = sum(p.numel() for p in trainable_params)
    fabric.print(f"Number of trainable parameters: {num_params:,}")
    trainable_param_names = [
        name for name, p in model.named_parameters() if p.requires_grad
    ]
    fabric.print(f"Trainable parameter names: {trainable_param_names}")
    num_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    fabric.print(f"Number of non trainable parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    fabric.print("hi-4")
    model, optimizer = fabric.setup(model, optimizer)

    fabric.print("hi-3")

    fabric.seed_everything(1337 + fabric.global_rank)

    fabric.print("hi-2")

    train_time = time.time()
    fabric.print("hi-1")
    train(
        fabric,
        model,
        optimizer,
        dataset,
        checkpoint_dir,
        out_dir,
        speed_monitor,
    )
    fabric.print(f"Training time: {(time.time()-train_time):.2f}s")

    # Save the final checkpoint at the end of training
    save_path = out_dir / "lit_model_finetuned.pth"
    save_checkpoint(fabric, model, save_path)


def train(
    fabric: L.Fabric,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    train_data: DatasetDict,
    checkpoint_dir: Path,
    out_dir: Path,
    speed_monitor: SpeedMonitor,
) -> None:
    fabric.print("hi0")
    tokenizer = Tokenizer(checkpoint_dir)
    fabric.print("hi0.5")

    fabric.print("hi0.75")

    fabric.print("hi1")

    fabric.print(
        f"starting val loss: {validate(fabric, model, train_data['validation'], tokenizer):.4f}"
    )

    # with torch.device("meta"):
    #     meta_model = GPT(model.config, num_tokens_in_soft_prompt)
    #     x = torch.randint(0, 1, (micro_batch_size, model.config.block_size))
    #     measured_flops = measure_flops(meta_model, x)
    #     fabric.print(
    #         f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}"
    #     )
    #     del meta_model, x

    measured_flops = 69696969.0

    step_count = 0
    total_lengths = 0
    total_t0 = time.time()

    for iter_num in range(max_iters):
        if step_count <= warmup_steps:
            # linear warmup
            lr = learning_rate * step_count / warmup_steps
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        iter_t0 = time.time()

        input_ids, targets = get_batch(
            fabric,
            train_data["train"],
            tokenizer,
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
            # this assumes that device FLOPs are the same and that all devices have the same batch size
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
                train_data["validation"],
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
            checkpoint_path = out_dir / f"iter-{iter_num:06d}-ckpt.pth"
            save_checkpoint(fabric, model, checkpoint_path)


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
        input_ids, targets = get_batch(fabric, val_dataset, tokenizer)

        logits = model(input_ids)
        loss = chunked_cross_entropy(logits, targets, chunk_size=0)
        losses[k] = loss.item()

        # Target generating 5 examples.
        if k % (eval_iters // 5) == 0:
            sample = strip_right_pad(input_ids[0])
            target = strip_right_pad(targets[0])

            colons = torch.nonzero(sample == 29901, as_tuple=False).squeeze()

            # Get the second occurrence index, if it exists
            second_index = colons[1] + 1 if len(colons) > 1 else None

            max_new_tokens = 10

            print(f"INPUT: {tokenizer.decode(sample[:second_index])}")
            output = sample_model(
                model,
                idx=sample[:second_index],
                max_new_tokens=max_new_tokens,
                temperature=0.01,
            )
            print(
                f"OUTPUT (decoded, tkns):",
                tokenizer.decode(output[-max_new_tokens:]),
            )

            target[target == -1] = 0

            print(
                f"TARGET (decoded, tkns):",
                tokenizer.decode(target),
            )
            print("\n\n")

    val_loss = losses.mean()

    model.train()
    return val_loss.item()


def get_batch(
    fabric: L.Fabric,
    data: Dataset,
    tokenizer: Tokenizer,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # TODO: This currently doesn't work with micro_batch_size > 1
    ix = torch.randint(len(data), (micro_batch_size,))

    # TODO: make it so that this can handle <MASK></MASK> system and <SOFT_PROMPT> (can we do via the model's tokenizer?)
    # TODO: preinit soft prompt

    raw_seqs = [
        torch.cat(
            (
                tokenizer.encode(
                    # TODO: dont just grab first 1k token lols
                    format_prompt(
                        data[i.item()]["Problem"], data[i.item()]["Solution"]
                    ),
                ).type(torch.int64),
            ),
            dim=0,
        )
        for i in ix
    ]

    # TODO: mask the vicuna prompt

    # TODO: THIS DOESN'T MASK OUT THE SOFT PROMPT, WE SHOULD ADD AN EXTRA DS FIELD FOR THAT???

    input_ids = [seq[:-1] for seq in raw_seqs]
    labels = [seq[1:] for seq in raw_seqs]

    # TODO can compute this number from base system prompt along with the other template items!!!
    num_masked_tokens = (
        num_tokens_in_soft_prompt + 52  # todo: double check this number lol
    )  # assumes soft prompt is not the first thing, otherwise u'd have to sub 1

    labels = [
        torch.cat(
            (
                torch.full((num_masked_tokens,), -1, dtype=torch.int64),
                label[num_masked_tokens:],
            )
        )
        for label in labels
    ]

    max_len = max(len(s) for s in input_ids)

    x = torch.stack([pad_right(x, pad_id=pad_tkn, pad_to=max_len) for x in input_ids])
    y = torch.stack([pad_right(x, pad_id=ignored_tkn, pad_to=max_len) for x in labels])

    if fabric.device.type == "cuda" and x.device.type == "cpu":
        x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
    else:
        x, y = fabric.to_device((x, y))
    return x, y


def save_checkpoint(fabric, model, file_path: Path):
    fabric.print(f"Saving weights to {str(file_path)!r}")
    fabric.save(file_path, {"model": model})


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(setup)
