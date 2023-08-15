import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import lightning as L
import torch
from datasets import load_dataset, DatasetDict, Dataset
from lightning.fabric.strategies import FSDPStrategy

# support running without installing as a package
# REMEMBER TO IMPORT ALL LOCAL DEPS AFTER THIS
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate.base import generate


from bistro.model import GPT, Config, Block
from lit_gpt.tokenizer import Tokenizer
from lit_gpt.utils import (
    lazy_load,
    check_valid_checkpoint_dir,
    step_csv_logger,
    chunked_cross_entropy,
)
from lit_gpt.speed_monitor import (
    SpeedMonitorFabric as SpeedMonitor,
    measure_flops,
    estimate_flops,
)

# from lit_gpt.scripts.prepare_alpaca import generate_prompt

eval_interval = 50
save_interval = 600
eval_iters = 100
log_interval = 1
devices = 1
# change this value to force a maximum sequence length
override_max_seq_length = None

# Hyperparameters
learning_rate = 3
batch_size = 64 / devices
micro_batch_size = 1
gradient_accumulation_iters = 3  # todo batch_size // micro_batch_size
assert gradient_accumulation_iters > 0
epoch_size = 50000  # train dataset size
num_epochs = 5
max_iters = num_epochs * (epoch_size // micro_batch_size) // devices
weight_decay = 0.02
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


def format_prompt(game: str) -> str:
    system_prompt = (
        "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, "
        "detailed, and polite answers to the user's questions. USER: {user_prompt} ASSISTANT: {game}"
    )
    formatted = system_prompt.format(
        user_prompt="Generate a game of chess at the Grandmaster level.", game=game
    )
    # print(formatted)
    return formatted


def setup(
    data_dir: Path = Path("data/chess"),
    checkpoint_dir: Path = Path("checkpoints/lmsys/vicuna-13b-v1.3"),
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

    logger = step_csv_logger(
        out_dir.parent, out_dir.name, flush_logs_every_n_steps=log_interval
    )
    fabric = L.Fabric(
        devices=devices, strategy=strategy, precision=precision, loggers=logger
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
        model = GPT(config)
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
    model, optimizer = fabric.setup(model, optimizer)

    fabric.seed_everything(1337 + fabric.global_rank)

    train_time = time.time()
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
    tokenizer = Tokenizer(checkpoint_dir)
    max_seq_length, longest_seq_length, longest_seq_ix = get_max_seq_length(
        train_data["train"]
    )

    fabric.print(
        f"starting val loss: {validate(fabric, model, train_data['validation'], tokenizer):.4f}"
    )

    with torch.device("meta"):
        meta_model = GPT(model.config)
        # estimated is too much of an optimistic estimate, left just for reference
        estimated_flops = estimate_flops(meta_model) * micro_batch_size
        fabric.print(
            f"Estimated TFLOPs: {estimated_flops * fabric.world_size / 1e12:.2f}"
        )
        x = torch.randint(0, 1, (micro_batch_size, model.config.block_size))
        measured_flops = measure_flops(meta_model, x)
        fabric.print(
            f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}"
        )
        del meta_model, x

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
            longest_seq_ix if iter_num == 0 else None,
        )

        is_accumulating = (iter_num + 1) % gradient_accumulation_iters != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids, max_seq_length=max_seq_length)
            # print(logits)
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

        if k == 0:
            og_sample = input_ids[0]
            og_target = targets[0]

            for i in range(1, 5):
                sample = og_sample[:-i]
                print(f"DECODED SAMPLE: |{tokenizer.decode(sample)}|")
                max_returned_tokens = len(sample) + 1
                output = generate(
                    model,
                    idx=sample,
                    max_returned_tokens=max_returned_tokens,
                    max_seq_length=max_returned_tokens,
                    temperature=0.01,
                )
                print(
                    f"PREDICTED TOKEN: |{tokenizer.decode(output[-1])}| ({output[-1]})"
                )
                print(
                    f"TARGET TOKEN: |{tokenizer.decode(og_target[-(i+1)])}| ({og_target[-(i+1)]})"
                )
                model.reset_cache()

    val_loss = losses.mean()

    model.reset_cache()

    model.train()
    return val_loss.item()


def get_batch(
    fabric: L.Fabric,
    data: Dataset,
    tokenizer: Tokenizer,
    longest_seq_ix: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data), (micro_batch_size,))

    if longest_seq_ix is not None:
        # force the longest sample at the beginning so potential OOMs happen right away
        ix[0] = longest_seq_ix

    raw_seqs = [
        torch.cat(
            (
                torch.tensor(([0] * 20), dtype=torch.int64),
                tokenizer.encode(
                    # TODO: dont just grab first 1k token lols
                    format_prompt(data[i.item()]["moves"][:1000])
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
    # replace the first 70 tokens in the label with -1
    labels = [
        torch.cat((torch.full((70,), -1, dtype=torch.int64), label[70:]))
        for label in labels
    ]

    max_len = max(len(s) for s in input_ids)

    def pad_right(x, pad_id):
        # pad right based on the longest sequence
        n = max_len - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
    y = torch.stack([pad_right(x, pad_id=-1) for x in labels])

    if fabric.device.type == "cuda" and x.device.type == "cpu":
        x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
    else:
        x, y = fabric.to_device((x, y))
    return x, y


def get_max_seq_length(data: Dataset) -> Tuple[int, int, int]:
    # find out the minimum max_seq_length required during fine-tuning (saves memory!)
    # todo: don't just grab the first 1k chars
    lengths = [len(d["moves"][:1000]) for d in data]
    max_seq_length = max(lengths)
    longest_seq_ix = lengths.index(max_seq_length)
    # support easy override at the top of the file
    return (
        override_max_seq_length
        if isinstance(override_max_seq_length, int)
        else max_seq_length,
        max_seq_length,
        longest_seq_ix,
    )


def save_checkpoint(fabric, model, file_path: Path):
    fabric.print(f"Saving weights to {str(file_path)!r}")
    fabric.save(file_path, {"model": model})


if __name__ == "__main__":
    # Uncomment this line if you see an error:
    # "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(setup)
