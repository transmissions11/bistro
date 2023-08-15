import json
import sys
import time
import warnings
from pathlib import Path
from typing import Optional, Literal

import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import torch.nn.functional as F

from bistro import GPT
from lit_gpt import Tokenizer, Config
from bistro.model import Block
from lit_gpt.utils import lazy_load, check_valid_checkpoint_dir


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    *,
    temperature: float = 1.0,
    eos_id: Optional[int] = None,
) -> torch.Tensor:

    """Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.

    The implementation of this function is modified from A. Karpathy's nanoGPT.

    Args:
        model: The model to use.
        idx: Tensor of shape (T) with indices of the prompt sequence.
        max_new_tokens: The maximum number of tokens to generate.
        temperature: Scales the predicted logits by 1 / temperature.
        eos_id: If specified, stop generating any more token once the <eos> token is triggered.
    """
    device, dtype = idx.device, idx.dtype

    # Create a tensor to hold the decoded tokens as we sample.
    decoded_tkns = torch.empty(0, device=device, dtype=dtype)

    for i in range(max_new_tokens):

        # Forward pass through the model.
        logits = model(torch.cat((idx, decoded_tkns)).unsqueeze(0))

        # Pluck the logits at the final step and scale by desired temperature.
        logits = logits[:, -1, :] / (
            temperature + 1e-10
        )  # +1e-10 as eps to avoid divide by zero

        # Apply softmax to convert logits to (normalized) probabilities.
        probs = F.softmax(logits, dim=-1)

        # Sample the next token, using [0] to remove the batch dim.
        next_tkn = torch.multinomial(probs, num_samples=1)[0]

        # Append the token to the running decoded sequence.
        decoded_tkns = torch.cat((decoded_tkns, next_tkn))

        # If the token is <|endoftext|>, we're done.
        if next_tkn.item() == eos_id:
            break

    return decoded_tkns


def main(
    prompt: str = "Hello, my name is",
    *,
    num_samples: int = 1,
    max_new_tokens: int = 50,
    top_k: int = 200,
    temperature: float = 0.8,
    checkpoint_dir: Path = Path("checkpoints/lmsys/vicuna-7b-v1.3"),
    strategy: str = "auto",
    devices: int = 1,
    precision: str = "bf16-true",
) -> None:
    """Generates text samples based on a pre-trained model and tokenizer.

    Args:
        prompt: The prompt string to use for generating the samples.
        num_samples: The number of text samples to generate.
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        checkpoint_dir: The checkpoint directory to load.
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

    model_file = "lit_model.pth"
    checkpoint_path = checkpoint_dir / model_file

    fabric.print(
        f"Loading model {str(checkpoint_path)!r} with {config.__dict__}",
        file=sys.stderr,
    )
    t0 = time.time()
    with fabric.init_module(empty_init=True):
        model = GPT(config)
    fabric.print(
        f"Time to instantiate model: {time.time() - t0:.02f} seconds.", file=sys.stderr
    )

    t0 = time.time()
    with lazy_load(checkpoint_path) as checkpoint:
        model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    fabric.print(
        f"Time to load the model weights: {time.time() - t0:.02f} seconds.",
        file=sys.stderr,
    )

    model.eval()
    model = fabric.setup_module(model)

    tokenizer = Tokenizer(checkpoint_dir)
    encoded = tokenizer.encode(prompt, device=fabric.device)
    print(f"ENCODED TOKENS: {encoded}")
    prompt_length = encoded.size(0)
    max_returned_tokens = prompt_length + max_new_tokens
    assert max_returned_tokens <= model.config.block_size, (
        max_returned_tokens,
        model.config.block_size,
    )  # maximum rope cache length

    L.seed_everything(1234)
    for i in range(num_samples):
        t0 = time.perf_counter()
        y = generate(
            model,
            encoded,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        t = time.perf_counter() - t0

        model.reset_cache()
        fabric.print(tokenizer.decode(y))
        tokens_generated = y.size(0) - prompt_length
        fabric.print(
            f"Time for inference {i + 1}: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec",
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
