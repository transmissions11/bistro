import torch

from model import GPT


def token_gradients(model: GPT, input_ids, input_slice, target_ids):
    # embed_weights = self.model.transformer.wte.weight

    # # TODO: separate slice of input_ids

    # one_hot = torch.zeros(
    #     input_ids.shape[0],
    #     embed_weights.shape[0],
    #     device=self.device,
    #     dtype=embed_weights.dtype,
    # )

    # one_hot.scatter_(
    #     1,
    #     input_ids.unsqueeze(1),
    #     torch.ones(
    #         one_hot.shape[0], 1, device=self.device, dtype=embed_weights.dtype
    #     ),
    # )

    # one_hot.requires_grad_()

    # input_embeds = (one_hot @ embed_weights).unsqueeze(0)

    # # now stitch it together with the rest of the embeddings
    # embeds = self.model.transformer.wte(input_ids.unsqueeze(0)).detach()
    # full_embeds = torch.cat(
    #     [
    #         embeds[:, : input_slice.start, :],
    #         input_embeds,
    #         embeds[:, input_slice.stop :, :],
    #     ],
    #     dim=1,
    # )

    # loss = self.compute_loss(targets, input_embeds=full_embeds)

    # # loss.backward()
    # self.manual_backward(loss)

    # grad = one_hot.grad.clone()

    embed_weights = model.transformer.wte.weight
    one_hot = torch.zeros(
        input_ids[input_slice].size(0),
        embed_weights.size(0),  # Vocab size.
        device=model.device,
        dtype=embed_weights.dtype,
    )
    one_hot.scatter_(
        1,
        input_ids[input_slice].unsqueeze(1),
        torch.ones(one_hot.size(0), 1, device=model.device, dtype=embed_weights.dtype),
    )
    one_hot.requires_grad_()

    # now stitch it together with the rest of the embeddings
    # TODO: can we re-use embed_weights?
    # TODO: do we actually need to detach given we've turned off grads for everything earlier?
    detached_embeds = model.transformer.wte(
        input_ids.unsqueeze(0)
    ).detach()  # Detaching to prevent updates during backpropagation

    full_embeds = torch.cat(
        [
            detached_embeds[:, : input_slice.start, :],
            (one_hot @ embed_weights).unsqueeze(0),
            detached_embeds[:, input_slice.stop :, :],
        ],
        dim=1,
    )

    logits = model(inputs_embeds=full_embeds).logits
    targets = input_ids[target_slice]
    loss = nn.CrossEntropyLoss()(logits[0, loss_slice, :], targets)

    # so my understanding is input slice is what part to treat as the control sequence,
    #  loss slice is basically a poor mans mask over the input, and targets is a poor mans mask to get targets all from one long input seq
    # can verify by just running

    loss.backward()

    # why are we grad norming
    grad = one_hot.grad.clone()
    grad = grad / grad.norm(dim=-1, keepdim=True)

    return grad
