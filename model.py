import torch
import torch.nn as nn

from transformers import SiglipVisionModel, SiglipVisionConfig


class MultiFrameSiglipClassifier(nn.Module):
    def __init__(
        self,
        model_id: str,
        num_ctx_frames: int,
        num_classes: int,
    ) -> None:
        super().__init__()

        self.num_ctx_frames = num_ctx_frames
        self.num_classes = num_classes

        self.model: SiglipVisionModel = SiglipVisionModel.from_pretrained(model_id)

        self.config: SiglipVisionConfig = self.model.config

        self.pos_embeddings = nn.Embedding(self.num_ctx_frames, self.config.hidden_size)

        self.classifier = nn.Linear(
            self.num_ctx_frames * self.config.hidden_size, self.num_classes
        )

    def forward(
        self,
        frames: torch.Tensor,  # [B, num_ctx_frames, C=3, H=hidden_size, W=hidden_size]
    ) -> torch.Tensor:

        frames = frames.view(
            -1, 3, self.config.hidden_size, self.config.hidden_size
        )  # [B*num_ctx_frames, C=3, H=hidden_size, W=hidden_size]

        x = self.model(frames).pooler_output  # [B*num_ctx_frames, hidden_size]

        x = x.view(
            -1, self.num_ctx_frames, self.config.hidden_size
        )  # [B, num_ctx_frames, hidden_size]

        x = x + self.pos_embeddings.weight  # [B, num_ctx_frames, hidden_size]

        x = self.projection(x.flatten(1))  # [B, hidden_size]

        return x
