import torch.nn as nn

from typing import Callable


def freeze_parameters(model: nn.Module, should_freeze: Callable[[str], bool]) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = not should_freeze(name)


def init_weights_optimally(module: nn.Module) -> None:
    """Meant to be used with `model.apply(init_weights)`."""
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
