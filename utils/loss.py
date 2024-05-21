import torch
import torch.nn.functional as F


def compute_loss(
    model: torch.nn.Module,
    *,  # Force keyword args to avoid confusion.
    inputs: torch.Tensor,
    labels: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    logits = model(inputs)
    print("logits.shape", logits.shape)
    print("labels.shape", labels.shape)
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        reduction=reduction,
    )
