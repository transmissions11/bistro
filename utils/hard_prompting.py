import torch

from typing import Optional

from lit_gpt import Tokenizer

from utils.loss import compute_loss
from utils.padding import pad_collate_fn

from model import GPT


def get_non_ascii_tkns(tokenizer: Tokenizer):
    """Returns a tensor of all non-ASCII token ids in the tokenizer's vocabulary."""

    def is_ascii(s):
        return s.isascii() and s.isprintable()

    return torch.tensor(
        [
            i
            for i in range(3, tokenizer.vocab_size)
            if not is_ascii(tokenizer.decode(torch.tensor(i)))
        ]
    )  # (num_non_ascii_tkns)


def get_hard_prompt_gradients(
    model: GPT,
    *,  # Force keyword arguments.
    current_hard_prompt: torch.Tensor,  # (num_hard_prompt_tkns)
    hard_prompt_tkn: int,
    input_ids: torch.Tensor,  # (b = 1, t)
    target_ids: torch.Tensor,  # (b = 1, t)
) -> torch.Tensor:
    """Calculates the gradients of the hard prompt with respect to the loss."""

    input_ids = input_ids.squeeze(0)  # (t)

    embed_weights = model.transformer.wte.weight  # (vocab_size, emb_dim)

    # Find the position of the hard prompt template in input_ids.
    hard_prompt_positions = torch.where(input_ids == hard_prompt_tkn)[0]
    hard_prompt_start_pos = hard_prompt_positions[0].item()
    hard_prompt_end_pos = hard_prompt_positions[-1].item()

    # Ensure that the hard prompt template is the same length as the current hard prompt.
    assert hard_prompt_end_pos - hard_prompt_start_pos + 1 == current_hard_prompt.size(
        0
    ), "mismatch between the calculated hard prompt length and current hard prompt length"

    # Create a one hot tensor of the hard prompt — (num_hard_prompt_tkns, vocab_size)
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

    # Detached to prevent updates during backpropagation — (b = 1, t, emb_dim)
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
    )  # () — scalar loss

    loss.backward()

    grad = one_hot.grad.clone()  # (num_hard_prompt_tkns, vocab_size)

    return grad / grad.norm(dim=-1, keepdim=True)  # (num_hard_prompt_tkns, vocab_size)


def create_hard_prompt_candidates(
    *,  # Force keyword arguments.
    current_hard_prompt: torch.Tensor,  # (num_hard_prompt_tkns)
    hard_prompt_grads: torch.Tensor,  # (num_hard_prompt_tkns, vocab_size)
    tokenizer: Tokenizer,
    num_candidates: int,
    topk: int,
    # Can be used to use only ASCII tokens, for example.
    not_allowed_tokens: Optional[torch.Tensor] = None,  # (num_not_allowed_tokens)
) -> torch.Tensor:
    """
    Creates a batch of hard prompt candidates by sampling randomly from the top-k tokens.

    The top-k tokens are determined by the hard prompt gradients. Should be used with
    filter_hard_prompt_candidates to ensure the length of the candidates doesn't explode.
    """

    # Ensure that (num_hard_prompt_tkns * topk) > num_candidates, as
    # otherwise we'd start repeating candidates and be wasting compute.
    assert (
        current_hard_prompt.size(0) * topk > num_candidates
    ), f"num_candidates ({num_candidates}) should be < (num_hard_prompt_tkns ({current_hard_prompt.size(0)}) * topk ({topk}))"

    # Set the gradients of not allowed tokens to infinity.
    if not_allowed_tokens is not None:
        hard_prompt_grads[:, not_allowed_tokens] = float("inf")

    # Create a (num_hard_prompt_tkns, topk) tensor of the top-k token ids that would decrease loss for each hard prompt token.
    top_tkns = (-hard_prompt_grads).topk(topk, dim=1).indices

    # Create a (num_candidates, num_hard_prompt_tkns) tensor of the current hard prompt.
    candidates = current_hard_prompt.repeat(num_candidates, 1)

    # Generate a (num_candidates) tensor of an index to replace in each row.
    new_token_pos = torch.arange(
        0,
        len(current_hard_prompt),
        len(current_hard_prompt) / num_candidates,
        device=hard_prompt_grads.device,
        # Need to run arange in a higher precision dtype before casting to int to
        # avoid rounding issues which result in indices >= len(current_hard_prompt).
        dtype=torch.float32,
    ).type(torch.int64)

    # Generate a (num_candidates, 1) tensor of token ids to replace each new_token_pos index with.
    new_token_val = torch.gather(
        top_tkns[new_token_pos],  # (num_candidates, topk)
        1,
        # (num_candidates, 1) — Random index from 0 to topk for each candidate.
        torch.randint(0, topk, (num_candidates, 1), device=hard_prompt_grads.device),
    )

    # (num_candidates, num_hard_prompt_tkns) — replace the new_token_pos
    # index for each candidate with the corresponding new_token_val token.
    raw_hard_prompt_candidates = candidates.scatter_(
        1, new_token_pos.unsqueeze(-1), new_token_val
    )

    filtered_candidates = []  # Create a list to store the filtered candidates.

    for candidate in raw_hard_prompt_candidates:
        # Ensure the candidate is not the same as the current hard prompt.
        if torch.equal(candidate, current_hard_prompt):
            continue

        # Decode and encode it again, to ensure we can use the
        # hard prompt on a model that only accepts text inputs.
        reencoded_candidate = tokenizer.encode(
            tokenizer.decode(candidate),
            # bos/eos=False because the candidates
            # will be in the middle of the sequence.
            bos=False,
            eos=False,
        ).to(raw_hard_prompt_candidates)

        # Ensure the candidate is the same length after decoding and encoding.
        if candidate.size(0) == reencoded_candidate.size(0):
            filtered_candidates.append(reencoded_candidate)

    # If the number of filtered candidates is less than the number of hard
    # prompt candidates, pad the list with the last candidate and return.
    return torch.stack(
        filtered_candidates
        + [filtered_candidates[-1]]
        * (len(raw_hard_prompt_candidates) - len(filtered_candidates))
    )


@torch.inference_mode()
def test_hard_prompt_candidates(
    model: GPT,
    *,  # Force keyword arguments.
    candidate_batch_size: int,
    hard_prompt_candidates: torch.Tensor,  # (num_candidates, num_hard_prompt_tkns)
    hard_prompt_tkn: int,
    input_ids: torch.Tensor,  # (b = 1, t)
    target_ids: torch.Tensor,  # (b = 1, t)
) -> torch.Tensor:
    """Returns the minimum loss and the index of the hard prompt candidate that yields the lowest loss when inserted into the input_ids sequence."""

    input_ids = input_ids.squeeze(0)  # (t)
    target_ids = target_ids.squeeze(0)  # (t)

    # Find the position of the hard prompt template in input_ids.
    hard_prompt_positions = torch.where(input_ids == hard_prompt_tkn)[0]
    hard_prompt_start_pos = hard_prompt_positions[0].item()
    hard_prompt_end_pos = hard_prompt_positions[-1].item()

    # Create a list to store input_id/target_id pairs for each candidate.
    # This is called a "mega batch" as we'll be splitting this into smaller
    # batches of size candidate_batch_size when actually testing them later on.
    mega_batch = []

    for candidate in hard_prompt_candidates:
        # Replace the hard prompt in the input sequence with the candidate.
        new_input_ids = input_ids.clone()
        new_input_ids[hard_prompt_start_pos : hard_prompt_end_pos + 1] = candidate

        # Add the new input/target pair to the mega batch.
        mega_batch.append({"inputs": new_input_ids, "targets": target_ids})

    # Pad the sequences and group everything into 2 tensors: inputs and targets.
    # NOTE: We don't actually need to pad since we're using the same input_ids
    # for each candidate, but we may need to in the future to support using
    # multiple input_ids, so leaving this in here for future compatibility.
    collated_mega_batch = pad_collate_fn(mega_batch)

    # Split the mega batch into smaller batches of size candidate_batch_size.
    input_batches, target_batches = (
        # (num_candidates, t) -> (num_candidates / candidate_batch_size, candidate_batch_size, t)
        torch.stack(collated_mega_batch["inputs"].split(candidate_batch_size, dim=0)),
        torch.stack(collated_mega_batch["targets"].split(candidate_batch_size, dim=0)),
    )

    # Compute and concentrate the losses for each candidate back into a single tensor.
    losses = torch.cat(
        [
            # compute_loss -> (candidate_batch_size * t)
            # .view(...) -> (candidate_batch_size, t)
            compute_loss(
                model,
                input_ids=inputs,
                target_ids=targets,
                reduction="none",
            ).view(targets.size(0), -1)
            for inputs, targets in zip(input_batches, target_batches)
        ],
    )  # [(candidate_batch_size, t), ...] -> (num_candidates, t)

    # Ignore losses of 0, as they are due to padding, return the mean of the rest.
    return losses[losses != 0].view(losses.size(0), -1).mean(dim=-1)  # (num_candidates)
