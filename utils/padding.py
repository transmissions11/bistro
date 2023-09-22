import torch

from typing import List, Dict

# Note: pad_tkn could be anything, since we pad
# right in effect the model never even sees them.
ignored_tkn, pad_tkn = -1, 0  # Some special tokens.


def pad_right(x: torch.Tensor, pad_id: int, pad_to: int) -> torch.Tensor:
    # TODO: should we be using len(x) here?
    return torch.nn.functional.pad(
        x, (0, pad_to - len(x)), mode="constant", value=pad_id
    )


def strip_right_pad(tensor):
    return tensor[
        : torch.max(torch.where((tensor != pad_tkn) & (tensor != ignored_tkn))[0]) + 1
    ]


# TODO: can we get the batch inputs to be in a better format that requires less work? wtf, look at HF pytorch lit example
# TODO: Detemrine x's type
def pad_collate_fn(batch: List[Dict[str, torch.Tensor]]):
    print("START batch: ", batch)

    # Find the maximum length of 'inputs' in the batch
    # TODO: better way to do this? can we do it automatically
    # TODO: Should we pad to a power of 8? (https://github.com/Lightning-AI/lit-gpt/pull/123/files)
    max_len = max([len(item["inputs"]) for item in batch])

    # Pad 'inputs' and 'targets' in each item in the batch
    inputs = [pad_right(item["inputs"], pad_tkn, max_len) for item in batch]
    targets = [pad_right(item["targets"], pad_tkn, max_len) for item in batch]

    # Combine 'inputs' and 'targets' into a single dictionary
    batch = {"inputs": torch.stack(inputs), "targets": torch.stack(targets)}

    print("END batch: ", batch)

    return batch
