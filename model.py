import torch
import torch.nn as nn

from transformers import SiglipVisionModel, SiglipVisionConfig


class SiglipClassifier(nn.Module):
    def __init__(
        self,
        model_id: str,
        num_classes: int,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes

        self.model: SiglipVisionModel = SiglipVisionModel.from_pretrained(model_id)

        self.config: SiglipVisionConfig = self.model.config

        self.classification_head = nn.Linear(self.config.hidden_size, self.num_classes)
        nn.init.lecun_normal_(self.classification_head.weight)
        if self.classification_head.bias is not None:
            nn.init.zeros_(self.classification_head.bias)

    def forward(
        self,
        frames: torch.Tensor,  # [B, C=3, H=image_size, W=image_size]
    ) -> torch.Tensor:

        x = self.model(frames).last_hidden_state  # [B, hidden_size]

        x = torch.mean(x[:, 1:, :], dim=1)  # TODO: why skip first token?

        x = self.classification_head(x)  # [B, num_classes]

        return x
