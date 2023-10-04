import torch

from utils.loss import compute_loss

from model import GPT

hard_prompt_template_tkn = (
    -1
)  # TODO (https://github.com/transmissions11/bistro/blob/soft-prompting/lit_model.py)


def token_gradients(model: GPT, input_ids: torch.Tensor, target_ids: torch.Tensor):
    embed_weights = model.transformer.wte.weight

    # find the position of the first occurrence of the soft_prompt_tkn in idx
    hard_prompt_positions = torch.where(input_ids == hard_prompt_template_tkn)
    hard_prompt_start_pos = hard_prompt_positions[0]
    hard_prompt_end_pos = hard_prompt_positions[-1]

    one_hot = torch.zeros(
        # The length of the hard prompt.
        (hard_prompt_end_pos - hard_prompt_start_pos + 1),
        embed_weights.size(0),  # Vocab size.
        device=input_ids.device,
        dtype=embed_weights.dtype,
    )
    one_hot.scatter_(
        1,
        input_ids[hard_prompt_start_pos : hard_prompt_end_pos + 1].unsqueeze(1),
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

    # so my understanding is input slice is what part to treat as the control sequence,
    #  loss slice is basically a poor mans mask over the input, and targets is a poor mans mask to get targets all from one long input seq
    # can verify by just running

    loss.backward()

    grad = one_hot.grad.clone()

    return grad / grad.norm(dim=-1, keepdim=True)
