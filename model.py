from typing import Optional, Tuple

import torch
import torch.nn as nn
from lit_gpt.config import Config
from lit_gpt.model import Block, build_rope_cache


class GPT(nn.Module):
    def __init__(self, config: Config, soft_prompt_tkn, num_soft_prompt_tkns) -> None:
        super().__init__()

        assert config.padded_vocab_size is not None
        self.config = config

        #############################################################################

        self.soft_prompt_tkn = soft_prompt_tkn
        self.num_soft_prompt_tkns = num_soft_prompt_tkns
        self.soft_prompt = nn.Parameter(
            # TODO: Allow init-ing this with some reasonable starting point.
            torch.randn(num_soft_prompt_tkns, config.n_embd)
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
        (B, T), block_size = idx.shape, self.config.block_size

        assert block_size >= T, f"[!] seq of len {T} exceeds block_size of {block_size}"

        # pass input tokens through the embedding layer
        x = self.transformer.wte(idx)  # (b, t, n_embd)

        #############################################################################

        # find the position of the first occurrence of the soft_prompt_tkn in idx
        soft_prompt_start_pos = torch.where(idx == self.soft_prompt_tkn)[1][0]

        # starting at soft_prompt_start_pos, replace num_tokens_in_soft_prompt tokens with the soft prompt
        x[
            :,
            soft_prompt_start_pos : soft_prompt_start_pos + self.num_soft_prompt_tkns,
        ] = self.soft_prompt

        #############################################################################

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

        for block in self.transformer.h:
            x, *_ = block(x, (cos, sin), block_size)  # (b, t, n_embd)

        x = self.transformer.ln_f(x)  # (b, t, n_embd)

        return self.lm_head(x)  # (b, t, vocab_size)
