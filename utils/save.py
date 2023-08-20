from pathlib import Path

import lightning as L
from torch import nn


def save_checkpoint(fabric: L.Fabric, model: nn.Module, file_path: Path):
    fabric.print(f"Saving model to {str(file_path)!r}...")
    fabric.save(file_path, {"model": model})
