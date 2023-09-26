import torch

from pathlib import Path

from lit_model import LitModel


from utils.inference import inference_model

from model import GPT

device = "cuda"


# TODO: why doesnt star work
def main(*, checkpoint: Path, temperature: float = 0.7, interactive: bool = False):
    print(checkpoint, temperature, interactive)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)
