from typing import Dict, List, Tuple

from torch import nn


def get_param_breakdown(model: nn.Module) -> Dict[str, List[Tuple[str, int]] or int]:
    trainable_param_info, non_trainable_param_info = [], []

    for name, p in model.named_parameters():
        if p.requires_grad:
            trainable_param_info.append((name, p.numel()))
        else:
            non_trainable_param_info.append((name, p.numel()))

    return {
        "trainable_param_names": [name for name, _ in trainable_param_info],
        "num_trainable_params": sum(numel for _, numel in trainable_param_info),
        "non_trainable_param_names": [name for name, _ in non_trainable_param_info],
        "num_non_trainable_params": sum(numel for _, numel in non_trainable_param_info),
    }


def mark_only_soft_prompt_as_trainable(model: nn.Module) -> None:
    """Sets `requires_grad=False` for all non-soft-prompt weights."""
    for name, param in model.named_parameters():
        param.requires_grad = "soft_prompt" in name
