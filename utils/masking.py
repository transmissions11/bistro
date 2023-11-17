import torch

from lit_gpt import Tokenizer

from utils.padding import ignored_tkn
from utils.tensors import find_subtensor_end


def mask_before_inclusive(
    delimiter: str, seq: torch.Tensor, tokenizer: Tokenizer
) -> torch.Tensor:
    """Replace all tokens before delimiter with ignored_tkn."""

    # bos/eos=False because the delimiter might be in the middle of the sequence.
    idx = find_subtensor_end(seq, tokenizer.encode(delimiter, bos=False, eos=False))

    print("SEQ", seq, "IDX", idx)

    return torch.cat(
        (
            torch.full((idx + 1,), ignored_tkn, dtype=seq.dtype),
            seq[idx + 1 :],
        )
    )
