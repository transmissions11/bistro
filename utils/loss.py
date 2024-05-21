import torch
import torch.nn.functional as F

from model import SiglipClassifier


def compute_loss(
    model: SiglipClassifier,
    *,  # Force keyword args to avoid confusion.
    inputs: torch.Tensor,
    labels: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    logits = model(inputs)
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        reduction=reduction,
    )
