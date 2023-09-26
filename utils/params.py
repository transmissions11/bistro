import torch.nn as nn

from typing import Callable


def freeze_parameters(model: nn.Module, should_freeze: Callable[[str], bool]) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = not should_freeze(name)
