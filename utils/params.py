import torch.nn as nn


def mark_only_soft_prompt_as_trainable(model: nn.Module) -> None:
    """Sets `requires_grad=False` for all non-soft-prompt weights."""
    for name, param in model.named_parameters():
        param.requires_grad = "soft_prompt" in name
