import torch

from utils.loss import compute_loss

from model import GPT


def token_gradients(
    model: GPT,
    *,  # Force keyword arguments.
    hard_prompt_tkn: int,
    current_hard_prompt: torch.Tensor,  # (num_hard_prompt_tkns)
    input_ids: torch.Tensor,  # (b = 1, t)
    target_ids: torch.Tensor,  # (b = 1, t)
    tokenizer,
):
    input_ids = input_ids.squeeze(0)  # (t)

    embed_weights = model.transformer.wte.weight

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

    # now stitch it together with the rest of the embeddings
    # TODO: can we re-use embed_weights?
    # TODO: do we actually need to detach given we've turned off grads for everything earlier?
    detached_embeds = model.transformer.wte(
        input_ids.unsqueeze(0)
    ).detach()  # Detaching to prevent updates during backpropagation

    input_embs = torch.cat(
        [
            # Everything before the hard prompt.
            detached_embeds[:, :hard_prompt_start_pos, :],
            # The hard prompt, undetached w/ grads.
            (one_hot @ embed_weights).unsqueeze(0),
            # Everything after the hard prompt.
            detached_embeds[:, hard_prompt_end_pos + 1 :, :],
        ],
        dim=1,
    )

    decoded = model.lm_head(input_embs)  # size -  torch.Size([1, 77, 32000])

    decoded_tokens = tokenizer.decode(decoded.argmax(dim=-1).squeeze(0))

    print(decoded_tokens)

    # print the shaped of transposed input weights
    print(
        model.transformer.wte.weight.shape,
        model.transformer.wte.weight.transpose(0, 1).shape,
    )
    loss = compute_loss(
        model,
        input_embs=input_embs,
        target_ids=target_ids,
    )

    # so my understanding is input slice is what part to treat as the control sequence,
    #  loss slice is basically a poor mans mask over the input, and targets is a poor mans mask to get targets all from one long input seq
    # can verify by just running

    loss.backward()

    grad = one_hot.grad.clone()

    return grad / grad.norm(dim=-1, keepdim=True)
