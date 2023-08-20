import torch

# Note: pad_tkn could be anything, since we pad
# right in effect the model never even sees them.
ignored_tkn, pad_tkn = -1, 0  # Some special tokens.


def pad_right(x, pad_id, pad_to):
    return torch.cat((x, torch.full((pad_to - len(x),), pad_id, dtype=x.dtype)))


def strip_right_pad(tensor):
    return tensor[
        : torch.max(torch.where((tensor != pad_tkn) & (tensor != ignored_tkn))[0]) + 1
    ]
