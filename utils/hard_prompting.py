import torch

from typing import Optional

from utils.loss import compute_loss

from model import GPT


def token_gradients(
    model: GPT,
    *,  # Force keyword arguments.
    hard_prompt_tkn: int,
    current_hard_prompt: torch.Tensor,  # (num_hard_prompt_tkns)
    input_ids: torch.Tensor,  # (b = 1, t)
    target_ids: torch.Tensor,  # (b = 1, t)
):
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


def sample_hard_prompt(
    hard_prompt_tkns: torch.Tensor,
    grad: torch.Tensor,
    batch_size: int,
    topk: int = 256,
    not_allowed_tokens: Optional[torch.Tensor] = None,
):
    if not_allowed_tokens is not None:
        grad[:, not_allowed_tokens.to(grad.device)] = float("inf")

    top_indices = (-grad).topk(topk, dim=1).indices
    hard_prompt_tkns = hard_prompt_tkns.to(grad.device)

    original_hard_prompt_tkns = hard_prompt_tkns.repeat(batch_size, 1)

    new_token_pos = torch.arange(
        0, len(hard_prompt_tkns), len(hard_prompt_tkns) / batch_size, device=grad.device
    ).type(torch.int64)

    new_token_val = torch.gather(
        top_indices[new_token_pos],
        1,
        torch.randint(0, topk, (batch_size, 1), device=grad.device),
    )

    new_hard_prompt_tkns = original_hard_prompt_tkns.scatter_(
        1, new_token_pos.unsqueeze(-1), new_token_val
    )

    print("SHAPE", new_hard_prompt_tkns.shape)
    print(f"{batch_size=}, {topk=}, {not_allowed_tokens=}")

    return new_hard_prompt_tkns
