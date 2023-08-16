from typing import Optional, Tuple, Any

import torch
import torch.nn as nn

from typing_extensions import Self

from lit_gpt.config import Config
from lit_gpt.rmsnorm import RMSNorm
from lit_gpt.model import Block, build_rope_cache

RoPECache = Tuple[torch.Tensor, torch.Tensor]


class GPT(nn.Module):
    def __init__(self, config: Config, num_tokens_in_soft_prompt) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config

        self.num_tokens_in_soft_prompt = num_tokens_in_soft_prompt
        self.soft_prompt = nn.Parameter(
            torch.randn(num_tokens_in_soft_prompt, config.n_embd)
        )

        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
                ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
            )
        )
        self.rope_cache: Optional[RoPECache] = None

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
            module.eps = self.config.norm_eps
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)
            module.eps = self.config.norm_eps

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
            self.rope_cache = self.build_rope_cache(idx)

        cos, sin = self.rope_cache
        cos = cos[:T]
        sin = sin[:T]

        # forward the token indexes through the embedding layer
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)

        # replace the first num_tokens_in_soft_prompt embs of each batch with the soft prompt embs
        x[:, : self.num_tokens_in_soft_prompt] = self.soft_prompt  # (b, t, n_embd)

        for block in self.transformer.h:
            x, *_ = block(x, (cos, sin), block_size)  # (b, t, n_embd)

        x = self.transformer.ln_f(x)  # (b, t, n_embd)

        return self.lm_head(x)  # (b, t, vocab_size)

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def build_rope_cache(self, idx: torch.Tensor) -> RoPECache:
        return build_rope_cache(
            seq_len=self.config.block_size,
            n_elem=int(self.config.rotary_percentage * self.config.head_size),
            dtype=torch.get_default_dtype(),
            device=idx.device,
            condense_ratio=self.config.condense_ratio,
        )
