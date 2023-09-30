import torch
import torch.nn.functional as F

from utils.padding import ignored_tkn

from model import GPT


def compute_loss(
    model: GPT, inputs: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    logits = model(input_ids=inputs)
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=ignored_tkn
    )
