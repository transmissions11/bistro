from typing import Tuple

import lightning as L
import torch
from datasets import Dataset
from lit_gpt import Tokenizer

from utils.padding import pad_right, pad_tkn, ignored_tkn
from utils.tensors import find_subtensor_end
from utils.vicuna import fmt_vicuna_input, VICUNA_END_OF_USER_PROMPT_SEQUENCE


def get_batch(
    fabric: L.Fabric,
    data: Dataset,
    tokenizer: Tokenizer,
    micro_batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    full_seqs = [
        tokenizer.encode(
            fmt_vicuna_input(data[i.item()]["prompt"], data[i.item()]["response"]),
        )
        for i in torch.randint(len(data), (micro_batch_size,))
    ]

    input_ids = [seq[:-1] for seq in full_seqs]
    labels = [
        # Mask everything before the assistant response.
        mask_before_inclusive(VICUNA_END_OF_USER_PROMPT_SEQUENCE, seq[1:], tokenizer)
        for seq in full_seqs
    ]

    max_len = max(len(s) for s in input_ids)

    x = torch.stack([pad_right(x, pad_id=pad_tkn, pad_to=max_len) for x in input_ids])
    y = torch.stack([pad_right(x, pad_id=ignored_tkn, pad_to=max_len) for x in labels])

    if fabric.device.type == "cuda" and x.device.type == "cpu":
        x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
    else:
        x, y = fabric.to_device((x, y))
    return x, y


def mask_before_inclusive(
    delimiter: str, seq: torch.Tensor, tokenizer: Tokenizer
) -> torch.Tensor:
    """Replace all tokens before delimiter with ignored_tkn."""

    # Find the last instance of delimiter in the sequence.
    idx = find_subtensor_end(seq, tokenizer.encode(delimiter))

    # Replace all tokens before and including the last instance of delimiter with ignored_tkn.
    return torch.cat(
        (
            torch.full((idx + 1,), ignored_tkn, dtype=seq.dtype),
            seq[idx + 1 :],
        )
    )
