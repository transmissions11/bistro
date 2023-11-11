import torch

from typing import Optional, Tuple

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
    )


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
    num_candidates: int,
    topk: int = 256,
    # Can be used to use only ASCII tokens, for example.
    not_allowed_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Creates a batch of hard prompt candidates by sampling randomly from the top-k tokens.

    The top-k tokens are determined by the hard prompt gradients. Should be used with
    filter_hard_prompt_candidates to ensure the length of the candidates doesn't explode.
    """

    # TODO: Build clean_hard_prompt_candidates into this function, and don't just
    # repeat the last candidate if after cleaning there are less than num_candidates.

    # Set the gradients of not allowed tokens to infinity.
    if not_allowed_tokens is not None:
        hard_prompt_grads[:, not_allowed_tokens] = float("inf")

    # Get the ids of the top-k tokens that would most decrease the loss.
    top_indices = (-hard_prompt_grads).topk(topk, dim=1).indices

    # Create a (num_candidates, num_hard_prompt_tkns) tensor of the current hard prompt.
    candidates = current_hard_prompt.repeat(num_candidates, 1)

    # Generate a (num_candidates) tensor of an index to replace in each row.
    new_token_pos = torch.arange(
        0,
        len(current_hard_prompt),
        len(current_hard_prompt) / num_candidates,
        device=hard_prompt_grads.device,
    ).type(torch.int64)

    # Generate a (num_candidates, 1) tensor of token ids to replace each new_token_pos index with.
    new_token_val = torch.gather(
        top_indices[new_token_pos],
        1,
        torch.randint(0, topk, (num_candidates, 1), device=hard_prompt_grads.device),
    )

    # Replace the new_token_pos index in each row with the new_token_val.
    return candidates.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)


def clean_hard_prompt_candidates(
    tokenizer: Tokenizer,
    *,  # Force keyword arguments.
    current_hard_prompt: torch.Tensor,  # (num_hard_prompt_tkns)
    hard_prompt_candidates: torch.Tensor,  # (num_candidates, num_hard_prompt_tkns)
) -> torch.Tensor:
    """
    Filters candidates that don't match the current hard prompt's length and cleans others
    so that candidates can be decoded and encoded again without changing their tokenization.

    Needed since encode(decode(tkns)) might yield a different tokenization.
    """

    filtered = []

    for candidate in hard_prompt_candidates:
        # Ensure the candidate is not the same as the current hard prompt.
        if torch.equal(candidate, current_hard_prompt):
            continue

        # Decode and encode it again, to ensure we can use the
        # hard prompt on a model that only accepts text inputs.
        reencoded_candidate = tokenizer.encode(tokenizer.decode(candidate)).to(
            hard_prompt_candidates  # Move to same device and dtype as candidates.
        )

        # Ensure the candidate is the same length after decoding and encoding.
        if candidate.size(0) == reencoded_candidate.size(0):
            filtered.append(reencoded_candidate)

    # If the number of filtered candidates is less than the number of hard
    # prompt candidates, pad the list with the last candidate and return.
    return torch.stack(
        filtered + [filtered[-1]] * (len(hard_prompt_candidates) - len(filtered))
    )


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
    # TODO: We don't actually need to pad since we're using the same input_ids
    # for each candidate, but we may need to in the future to support using
    # multiple input_ids, so leaving this in here for future compatibility.
    collated_mega_batch = pad_collate_fn(mega_batch)

    if torch.distributed.get_rank() == 0:
        for i in range(2):
            print(f"input_ids {i}:", collated_mega_batch["inputs"][i])

    # Split the mega batch into smaller batches of size candidate_batch_size.
    input_batches, target_batches = (
        torch.stack(collated_mega_batch["inputs"].split(candidate_batch_size, dim=0)),
        torch.stack(collated_mega_batch["targets"].split(candidate_batch_size, dim=0)),
    )

    losses = []  # Create a list to store the loss for each candidate.

    # TODO List comp instead of loop?
    for inputs, targets in zip(input_batches, target_batches):
        # TODO: profile why mem usage is 50% before even running compute_loss?
        # -> ah hmm probably just loading weights in and shit during configure_model???

        # TODO: use inference mode decorator or something instead?
        with torch.no_grad():
            # compute_loss -> (candidate_batch_size * t)
            # .view(...) -> (candidate_batch_size, t)
            loss = compute_loss(
                model,
                input_ids=inputs,
                target_ids=targets,
                reduction="none",
            ).view(targets.size(0), -1)

            losses.append(loss)

    losses = torch.cat(losses, dim=0)  # (num_candidates, t)

    # Ignore losses of 0, as they are due to padding, return the mean of the rest.
    return losses[losses != 0].view(losses.size(0), -1).mean(dim=-1)  # (num_candidates)
