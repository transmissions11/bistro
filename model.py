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

        self.classification_head = nn.Linear(
            self.num_ctx_frames * self.config.hidden_size, self.num_classes
        )

    def forward(
        self,
        frames: torch.Tensor,  # [B, num_ctx_frames, C=3, H=image_size, W=image_size]
    ) -> torch.Tensor:

        import ipdb

        ipdb.set_trace(
            cond=(
                (0 == torch.distributed.get_rank())
                if torch.distributed.is_initialized()
                else True
            )
        )

        frames = frames.view(
            -1, 3, self.config.image_size, self.config.image_size
        )  # [B*num_ctx_frames, 3, image_size, image_size]

        x = self.model(frames).pooler_output  # [B*num_ctx_frames, hidden_size]

        x = x.view(
            -1, self.num_ctx_frames, self.config.hidden_size
        )  # [B, num_ctx_frames, hidden_size]

        x = x + self.pos_embeddings.weight  # [B, num_ctx_frames, hidden_size]

        x = self.classification_head(x.flatten(1))  # [B, num_classes]

        return x
