import torch

# Note: pad_tkn could be anything, since we pad
# right in effect the model never even sees them.
ignored_tkn, pad_tkn = -1, 0  # Some special tokens.


# TODO: Should we pad to a power of 8? (https://github.com/Lightning-AI/lit-gpt/pull/123/files)
def pad_right(x: torch.Tensor, pad_id: int, pad_to: int) -> torch.Tensor:
    # TODO: should we be using len(x) here?
    return torch.nn.functional.pad(
        x, (0, pad_to - len(x)), mode="constant", value=pad_id
    )


def strip_right_pad(tensor):
    return tensor[
        : torch.max(torch.where((tensor != pad_tkn) & (tensor != ignored_tkn))[0]) + 1
    ]


# TODO: Detemrine x's type
def pad_collate_fn(x):
    # pad to the same length
    return pad_right(x, pad_tkn, 1000)  # TODO: set pad_to
