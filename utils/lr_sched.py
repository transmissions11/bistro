import math


# Note: This meant to be used with a LambdaLR lr_scheduler, which
# means the output is not the desired learning rate at step n, rather
# the ratio k such that k * max_learning_rate equals the output at step n.
# max_learning_rate is determined by your optimizer's lr parameter (e.g. AdamW)
def cosine_with_linear_warmup(
    step: int,
    # Hyperparameters:
    warmup_steps: int,
    min_lr_ratio: float,  # Ratio of max_learning_rate to anneal down to.
    total_steps: int,
) -> float:
    # 1) linear warmup for warmup_steps steps
    if step < warmup_steps:
        return step / warmup_steps
    # 2) if it > total_steps, return min learning rate ratio
    if step > total_steps:
        return min_lr_ratio
    # 3) in between, use cosine decay down to min learning rate ratio
    decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr_ratio + coeff * (1.0 - min_lr_ratio)
