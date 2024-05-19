import math


# via https://github.com/karpathy/nanoGPT
def cosine_with_linear_warmup(
    step: int,
    # Hyperparameters:
    warmup_steps: int,
    learning_rate: float,
    min_learning_rate: float,
    total_steps: int,
) -> float:
    # 1) linear warmup for warmup_steps steps
    if step < warmup_steps:
        return learning_rate * step / warmup_steps
    # 2) if it > total_steps, return min learning rate
    if step > total_steps:
        return min_learning_rate
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_learning_rate + coeff * (learning_rate - min_learning_rate)
