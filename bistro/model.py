from typing import Optional, Tuple

import torch
import torch.nn as nn

from lit_gpt.config import Config
from lit_gpt.model import Block, build_rope_cache


class GPT(nn.Module):
    def __init__(self, config: Config, num_tokens_in_soft_prompt) -> None:
        super().__init__()

        assert config.padded_vocab_size is not None
        self.config = config

        #############################################################################

        self.num_tokens_in_soft_prompt = num_tokens_in_soft_prompt
        self.soft_prompt = nn.Parameter(
            torch.randn(num_tokens_in_soft_prompt, config.n_embd)
        )

        #############################################################################

        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
                ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
            )
        )

        self.rope_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def forward(
        self,
        idx: torch.Tensor,
    ) -> torch.Tensor:
        B, T = idx.size()

        block_size = self.config.block_size

        assert (
            block_size >= T
        ), f"Cannot forward sequence of length {T}, block size is only {block_size}"

        if self.rope_cache is None:
            self.rope_cache = build_rope_cache(
                seq_len=block_size,
                n_elem=int(self.config.rotary_percentage * self.config.head_size),
                dtype=torch.get_default_dtype(),
                device=idx.device,
                condense_ratio=self.config.condense_ratio,
            )

        cos, sin = self.rope_cache
        cos = cos[:T]
        sin = sin[:T]

        # forward the token indexes through the embedding layer
        x = self.transformer.wte(idx)  # (b, t, n_embd)

        #############################################################################

        # swap the first num_tokens_in_soft_prompt embs of each batch for soft_prompt
        x[:, : self.num_tokens_in_soft_prompt] = self.soft_prompt  # (b, t, n_embd)

        #############################################################################

        for block in self.transformer.h:
            x, *_ = block(x, (cos, sin), block_size)  # (b, t, n_embd)

        x = self.transformer.ln_f(x)  # (b, t, n_embd)

        return self.lm_head(x)  # (b, t, vocab_size)
