import torch

from typing import Optional

from lit_gpt import Tokenizer

from utils.loss import compute_loss

from model import GPT


def get_hard_prompt_gradients(
    model: GPT,
    *,  # Force keyword arguments.
    current_hard_prompt: torch.Tensor,  # (num_hard_prompt_tkns)
    hard_prompt_tkn: int,
    input_ids: torch.Tensor,  # (b = 1, t)
    target_ids: torch.Tensor,  # (b = 1, t)
) -> torch.Tensor:
    """
    Calculates the gradients of the hard prompt with respect to the loss.
    """

    input_ids = input_ids.squeeze(0)  # (t)

    embed_weights = model.transformer.wte.weight  # (vocab_size, emb_dim)

    # find the position of the first occurrence of the hard_prompt_tkn in idx
    hard_prompt_positions = torch.where(input_ids == hard_prompt_tkn)[0]
    hard_prompt_start_pos = hard_prompt_positions[0].item()
    hard_prompt_end_pos = hard_prompt_positions[-1].item()

    # ensure that the hard prompt template is the same length as the current hard prompt
    assert hard_prompt_end_pos - hard_prompt_start_pos + 1 == current_hard_prompt.size(
        0
    ), "mismatch between the calculated hard prompt length and current hard prompt length"

    one_hot = torch.zeros(
        current_hard_prompt.size(0),
        embed_weights.size(0),  # Vocab size.
        device=input_ids.device,
        dtype=embed_weights.dtype,
    )
    one_hot.scatter_(
        1,
        current_hard_prompt.unsqueeze(1),
        torch.ones(
            one_hot.size(0), 1, device=input_ids.device, dtype=embed_weights.dtype
        ),
    )
    one_hot.requires_grad_()

    # Detached to prevent updates during backpropagation.
    detached_embeds = embed_weights[input_ids.unsqueeze(0)].detach()

    loss = compute_loss(
        model,
        input_embs=torch.cat(
            [
                # Everything before the hard prompt.
                detached_embeds[:, :hard_prompt_start_pos, :],
                # The hard prompt, undetached w/ grads.
                (one_hot @ embed_weights).unsqueeze(0),
                # Everything after the hard prompt.
                detached_embeds[:, hard_prompt_end_pos + 1 :, :],
            ],
            dim=1,
        ),
        target_ids=target_ids,
    )

    loss.backward()

    grad = one_hot.grad.clone()

    return grad / grad.norm(dim=-1, keepdim=True)


def create_hard_prompt_candidates(
    current_hard_prompt: torch.Tensor,  # (num_hard_prompt_tkns)
    hard_prompt_grads: torch.Tensor,  # (num_hard_prompt_tkns, vocab_size)
    *,  # Force keyword arguments.
    batch_size: int,
    topk: int = 256,
    # Can be used to use only ASCII tokens, for example.
    not_allowed_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Creates a batch of hard prompt candidates by sampling randomly from the top-k tokens.

    The top-k tokens are determined by the hard prompt gradients. Should be used with
    filter_hard_prompt_candidates to ensure the length of the candidates doesn't explode.
    """

    # Set the gradients of not allowed tokens to infinity.
    if not_allowed_tokens is not None:
        hard_prompt_grads[:, not_allowed_tokens] = float("inf")

    # Get the ids of the top-k tokens that would most decrease the loss.
    top_indices = (-hard_prompt_grads).topk(topk, dim=1).indices

    candidates_batch = current_hard_prompt.repeat(batch_size, 1)

    # Generate positions for new tokens in the hard prompt by creating a tensor of evenly spaced values.
    new_token_pos = torch.arange(
        0,
        len(current_hard_prompt),
        len(current_hard_prompt) / batch_size,
        device=hard_prompt_grads.device,
    ).type(torch.int64)

    # TODO: ADD A COMMENT AFTER PRINTING THIS
    print(new_token_pos)  # TODO: ADD A COMMENT AFTER PRINTING THIS
    # TODO: ADD A COMMENT AFTER PRINTING THIS

    new_token_val = torch.gather(
        top_indices[new_token_pos],
        1,
        torch.randint(0, topk, (batch_size, 1), device=hard_prompt_grads.device),
    )

    new_hard_prompt_tkns = candidates_batch.scatter_(
        1, new_token_pos.unsqueeze(-1), new_token_val
    )

    return new_hard_prompt_tkns


def filter_hard_prompt_candidates(
    tokenizer: Tokenizer,
    *,  # Force keyword arguments.
    current_hard_prompt: torch.Tensor,  # (num_hard_prompt_tkns)
    hard_prompt_candidates: torch.Tensor,  # (batch_size, num_hard_prompt_tkns)
) -> torch.Tensor:
    """
    Filters candidates that don't match the current hard prompt's length.

    This is due to the non-invertibility of tokenizers, where
    encode(decode(tkns)) might yield a different tokenization.
    """

    filtered = [
        candidate
        for candidate in hard_prompt_candidates
        # Ensure the candidate is not the same as the current hard prompt.
        if not torch.equal(candidate, current_hard_prompt)
        # Ensure the candidate is the same length after decoding and encoding.
        and candidate.size(0) == tokenizer.encode(tokenizer.decode(candidate)).size(0)
    ]

    # If the number of filtered candidates is less than the number of hard
    # prompt candidates, pad the list with the last candidate and return.
    return torch.stack(
        filtered + [filtered[-1]] * (len(hard_prompt_candidates) - len(filtered))
    )
