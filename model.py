import torch
import torch.nn as nn

from typing import Optional, Tuple

from lit_gpt.config import Config
from lit_gpt.model import Block


class GPT(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()

        assert config.padded_vocab_size is not None
        self.config = config

        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
                ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
            )
        )

    def forward(
        self,
        *,  # Force keyword args to avoid confusion.
        input_ids: Optional[torch.Tensor] = None,
        input_embs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_ids is not None and input_embs is not None:
            raise ValueError(
                "[!] you cannot specify both input_ids and input_embs at the same time"
            )
        elif input_embs is not None:
            x = input_embs
        elif input_ids is not None:
            x = self.embed(input_ids)  # (b, t, n_embd)
        else:
            raise ValueError("[!] you must specify either input_ids or input_embs")

        (B, T), block_size = x.shape[:2], self.config.block_size

        assert block_size >= T, f"[!] seq of len {T} exceeds block_size of {block_size}"

        if not hasattr(self, "cos"):
            cos, sin = self.rope_cache()  # Build the cache.
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)

        cos = self.cos[:T]
        sin = self.sin[:T]

        # TODO: del
        torch.set_printoptions(precision=15)
        # self = self.to(torch.get_default_dtype())
        # x = x.to(torch.get_default_dtype())

        print("PRE BLOCKS", x[0][0][0])

        for block in self.transformer.h:
            x, *_ = block(x, (cos, sin), block_size)  # (b, t, n_embd)
            print("POST BLOCK", x[0][0][0])

        x = self.transformer.ln_f(x)  # (b, t, n_embd)

        return self.lm_head(x)  # (b, t, vocab_size)

    def reset_caches(self):
        # TODO: WE NEED TO RESET THE ROPE CACHE?
        # also should we rename this to reset_parameters?
        # https://github.com/Lightning-AI/lit-gpt/blob/9cec81c38fc4df9ecb9250a0b64b1d8301cde259/lit_gpt/model.py#L57
        ...

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.transformer.wte(input_ids)

    def unembed(self, input_embs: torch.Tensor, argmax: bool = True) -> torch.Tensor:
        logits = input_embs @ self.transformer.wte.weight.T

        if argmax:
            return logits.argmax(dim=-1)  # (.., t)
        else:
            return logits  # (.., t, vocab_size)
