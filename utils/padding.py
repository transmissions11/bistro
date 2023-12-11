import torch

from typing import List, Dict

# Note: pad_tkn could be anything, since we pad
# right in effect the model never even sees them.
ignored_tkn, pad_tkn = -1, 0  # Some special tokens.


def pad_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    # batch = [{"inputs": (seq_len), "targets": (seq_len)}, ...]

    max_len = max([len(item["inputs"]) for item in batch])

    inputs, targets = [], []
    for item in batch:
        inputs.append(pad_right(item["inputs"], max_len))  # Defaults to pad_tkn.
        targets.append(pad_right(item["targets"], max_len, pad_id=ignored_tkn))

    # -> { "inputs": (batch_size, max_len), "targets": (batch_size, max_len) }
    return {"inputs": torch.stack(inputs), "targets": torch.stack(targets)}


def pad_right(x: torch.Tensor, pad_to: int, pad_id: int = pad_tkn) -> torch.Tensor:
    # x = (seq_len) -> (pad_to)
    return torch.nn.functional.pad(x, (0, pad_to - x.size(0)), value=pad_id)


def strip_right_pad(x: torch.Tensor, pad_id: int = pad_tkn) -> torch.Tensor:
    # x = (padded_len) -> (seq_len)
    return x[: torch.max(torch.where((x != pad_id) & (x != ignored_tkn))[0]) + 1]
