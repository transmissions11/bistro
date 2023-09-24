import torch
import torch.nn.functional as F

from typing import Optional


@torch.inference_mode()
def sample_model(
    model: torch.nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    *,
    temperature: float = 0.7,
    eos_id: Optional[int] = None,
) -> torch.Tensor:
    device, dtype = idx.device, idx.dtype

    # Create a tensor to hold the decoded tokens as we sample.
    decoded_tkns = torch.empty(0, device=device, dtype=dtype)

    for _ in range(max_new_tokens):
        # Forward pass through the model, unsqueeze to add a batch dimension.
        logits = model(input_ids=torch.cat((idx, decoded_tkns)).unsqueeze(0))

        # Pluck the logits at the final step and scale by desired temperature.
        logits = logits[:, -1, :] / (
            temperature + 1e-10
        )  # +1e-10 as eps to avoid divide by zero

        # Apply softmax to convert logits to (normalized) probabilities.
        probs = F.softmax(logits, dim=-1)

        # Sample the next token, using [0] to remove the batch dim.
        next_tkn = torch.multinomial(probs, num_samples=1)[0]

        # Append the token to the running decoded sequence.
        decoded_tkns = torch.cat((decoded_tkns, next_tkn))

        # If the token is <|endoftext|>, we're done.
        if next_tkn.item() == eos_id:
            break

    return decoded_tkns
