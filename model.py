import torch
import torch.nn as nn

from typing import Optional

from lit_gpt.config import Config
from lit_gpt.model import Block, build_rope_cache


class GPT(nn.Module):
    def __init__(self, config: Config, soft_prompt_tkn, num_soft_prompt_tkns) -> None:
        super().__init__()

        assert config.padded_vocab_size is not None
        self.config = config

        ############################################################################

        self.soft_prompt_tkn = soft_prompt_tkn
        self.num_soft_prompt_tkns = num_soft_prompt_tkns
        self.soft_prompt = nn.Parameter(
            # TODO: Allow initializing this with some reasonable starting sequence.
            torch.randn(num_soft_prompt_tkns, config.n_embd)
        )

        ############################################################################

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
        input_ids: Optional[torch.Tensor] = None,  # (b, t)
        input_embs: Optional[torch.Tensor] = None,  # (b, t, n_embd)
    ) -> torch.Tensor:
        if input_ids is not None and input_embs is not None:
            raise ValueError(
                "[!] you cannot specify both input_ids and input_embs at the same time"
            )
        elif input_embs is not None:
            x = input_embs  # (b, t, n_embd)
        elif input_ids is not None:
            # pass input tokens through the embedding layer
            x = self.embed(input_ids)  # (b, t, n_embd)

            ############################################################################

            # find the position of the first occurrence of the soft_prompt_tkn in idx
            soft_prompt_start_pos = torch.where(input_ids == self.soft_prompt_tkn)[1][0]

            # starting at soft_prompt_start_pos, replace num_tokens_in_soft_prompt tokens with the soft prompt
            x[
                :,
                soft_prompt_start_pos : soft_prompt_start_pos
                + self.num_soft_prompt_tkns,
            ] = self.soft_prompt

            ############################################################################
        else:
            raise ValueError("[!] you must specify either input_ids or input_embs")

        (_, T, _), block_size = x.shape, self.config.block_size

        assert block_size >= T, f"[!] seq of len {T} exceeds block_size of {block_size}"

        if not hasattr(self, "cos"):
            cos, sin = build_rope_cache(
                seq_len=block_size,
                n_elem=self.config.rope_n_elem,
                device=x.device,
                condense_ratio=self.config.rope_condense_ratio,
                base=self.config.rope_base,
            )
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)

        cos = self.cos[:T]
        sin = self.sin[:T]

        for block in self.transformer.h:
            x = block(x, cos, sin)  # (b, t, n_embd)

        x = self.transformer.ln_f(x)  # (b, t, n_embd)

        return self.lm_head(x)  # (b, t, vocab_size)

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.transformer.wte(input_ids)  # (..., t, n_embd)

    def unembed(self, input_embs: torch.Tensor, argmax: bool = True) -> torch.Tensor:
        logits = input_embs @ self.transformer.wte.weight.T

        if argmax:
            return logits.argmax(dim=-1)  # (..., t)
        else:
            return logits  # (..., t, vocab_size)
