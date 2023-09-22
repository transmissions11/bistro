import torch

from typing import List, Dict

# Note: pad_tkn could be anything, since we pad
# right in effect the model never even sees them.
ignored_tkn, pad_tkn = -1, 0  # Some special tokens.


def pad_collate_fn(batch: List[Dict[str, torch.Tensor]]):
    max_len = max([len(item["inputs"]) for item in batch])

    print(f"Max len: {max_len}")

    inputs, targets = [], []
    for item in batch:
        inputs.append(pad_right(item["inputs"], max_len))
        targets.append(pad_right(item["targets"], max_len))

    return {"inputs": torch.stack(inputs), "targets": torch.stack(targets)}


def pad_right(x: torch.Tensor, pad_to: int, pad_id: int = pad_tkn) -> torch.Tensor:
    return torch.nn.functional.pad(x, (0, pad_to - x.size(0)), value=pad_id)


def strip_right_pad(x: torch.Tensor, pad_id: int = pad_tkn) -> torch.Tensor:
    return x[: torch.max(torch.where((x != pad_id) & (x != ignored_tkn))[0]) + 1]
