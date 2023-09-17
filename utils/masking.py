import torch

from lit_gpt import Tokenizer

from utils.padding import ignored_tkn
from utils.tensors import find_subtensor_end


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
