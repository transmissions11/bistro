import torch
import torch.nn as nn

from transformers import SiglipVisionModel, SiglipVisionConfig
from transformers.models.siglip.modeling_siglip import lecun_normal_


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
        self._init_weights(self.classification_head)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            lecun_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        frames: torch.Tensor,  # [B, C=3, H=image_size, W=image_size]
    ) -> torch.Tensor:

        x = self.model(frames).last_hidden_state  # [B, hidden_size]

        x = torch.mean(x[:, 1:, :], dim=1)  # TODO: why skip first token?

        x = self.classification_head(x)  # [B, num_classes]

        return x
