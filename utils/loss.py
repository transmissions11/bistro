import torch
import torch.nn.functional as F

from typing import Optional

from utils.padding import ignored_tkn

from model import GPT


def compute_loss(
    model: GPT,
    *,  # Force keyword args to avoid confusion.
    target_ids: torch.Tensor,
    input_ids: Optional[torch.Tensor] = None,
    input_embs: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    model(input_ids=input_ids, input_embs=input_embs)
    # return F.cross_entropy(
    #     logits.view(-1, logits.size(-1)),
    #     target_ids.view(-1),
    #     ignore_index=ignored_tkn,
    #     reduction=reduction,
    # )
