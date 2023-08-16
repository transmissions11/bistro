import json
import sys
import time
import warnings
from pathlib import Path

import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate.base import generate
from bistro import GPT
from lit_gpt import Tokenizer, Config
from model import Block
from lit_gpt.utils import lazy_load, check_valid_checkpoint_dir


def main(
    prompt: str = "What food do lamas eat?",
    finetuned_path: Path = Path("out/full/chess/lit_model_finetuned.pth"),
    checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    strategy: str = "auto",
    devices: int = 1,
    precision: str = "bf16-true",
) -> None:
    """Generates a response based on a given instruction and an optional input.
    This script will only work with checkpoints from the instruction-tuned GPT model.
    See `finetune/full.py`.

    Args:
        prompt: The prompt/instruction (Alpaca style).
        input: Optional input (Alpaca style).
        finetuned_path: Path to the checkpoint with trained weights, which are the output of
            `finetune/full.py`.
        checkpoint_dir: The path to the checkpoint folder with pretrained GPT weights.
        max_new_tokens: The number of generation steps to take.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        strategy: Indicates the Fabric strategy setting to use.
        devices: How many devices to use.
        precision: Indicates the Fabric precision setting to use.
    """
    if strategy == "fsdp":
        strategy = FSDPStrategy(auto_wrap_policy={Block}, cpu_offload=False)
    fabric = L.Fabric(devices=devices, precision=precision, strategy=strategy)
    fabric.launch()

    check_valid_checkpoint_dir(checkpoint_dir)

    with open(checkpoint_dir / "lit_config.json") as fp:
        config = Config(**json.load(fp))

    fabric.print(
        f"Loading model {str(finetuned_path)!r} with {config.__dict__}",
        file=sys.stderr,
    )
    t0 = time.time()
    with fabric.init_module(empty_init=True):
        model = GPT(config)
    fabric.print(
        f"Time to instantiate model: {time.time() - t0:.02f} seconds.", file=sys.stderr
    )

    t0 = time.time()
    with lazy_load(finetuned_path) as checkpoint:
        model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    fabric.print(
        f"Time to load the model weights: {time.time() - t0:.02f} seconds.",
        file=sys.stderr,
    )

    model.eval()
    model = fabric.setup(model)

    tokenizer = Tokenizer(checkpoint_dir)
    encoded = tokenizer.encode(prompt, device=model.device)
    prompt_length = encoded.size(0)

    t0 = time.perf_counter()
    y = generate(
        model,
        encoded,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        eos_id=tokenizer.eos_id,
    )
    t = time.perf_counter() - t0

    output = tokenizer.decode(y)
    fabric.print(output)

    tokens_generated = y.size(0)
    fabric.print(
        f"\n\nTime for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec",
        file=sys.stderr,
    )
    if fabric.device.type == "cuda":
        fabric.print(
            f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB",
            file=sys.stderr,
        )


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore",
        message="ComplexHalf support is experimental and many operators don't support it yet",
    )
    CLI(main)
