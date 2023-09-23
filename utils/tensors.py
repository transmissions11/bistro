import torch

from typing import Optional


def find_subtensor_end(tensor: torch.Tensor, subset: torch.Tensor) -> Optional[int]:
    """Find the index of the last occurrence of subset in seq."""
    windows = tensor.unfold(0, len(subset), 1)
    idx = torch.where((windows == subset).all(dim=1))[0]
    return (idx[0] + len(subset) - 1).item() if idx.numel() else None
