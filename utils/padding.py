import torch

ignored_tkn, pad_tkn = -1, 0  # Relevant special tokens.


def pad_right(x, pad_id, pad_to):
    return torch.cat((x, torch.full((pad_to - len(x),), pad_id, dtype=x.dtype)))


def strip_right_pad(tensor):
    return tensor[
        : torch.max(torch.where((tensor != pad_tkn) & (tensor != ignored_tkn))[0]) + 1
    ]
