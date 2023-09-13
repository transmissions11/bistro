import torch
from torch.utils.data import DataLoader

import lightning.pytorch as L

from lit_gpt import Tokenizer

from datasets import load_dataset

from utils.masking import mask_before_inclusive
from utils.vicuna import VICUNA_END_OF_USER_PROMPT_SEQUENCE, fmt_vicuna_input


class LitDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        tokenizer: Tokenizer,
        num_soft_prompt_tkns: int,
        soft_prompt_tkn: str,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.num_soft_prompt_tkns = num_soft_prompt_tkns
        self.soft_prompt_tkn = soft_prompt_tkn

    def download_and_transform(self):
        def transform(x):
            seq = self.tokenizer.encode(
                fmt_vicuna_input(
                    f"{self.soft_prompt_tkn * self.num_soft_prompt_tkns} {x['prompt']}",
                    x["response"],
                )
            ).type(torch.int64)

            return {
                "input_ids": seq[:-1],
                # Mask everything before the assistant response.
                "targets": mask_before_inclusive(
                    VICUNA_END_OF_USER_PROMPT_SEQUENCE, seq[1:], self.tokenizer
                ),
            }

        return (
            load_dataset("parquet", data_dir=self.data_dir)
            .with_format("torch")
            .map(
                transform,
                remove_columns=["prompt", "response"],
                load_from_cache_file=False,  # TODO: Fix this.
                # num_proc=2,
                # We can force cache like this if needed.
                # new_fingerprint="t1"
            )
        )

    def prepare_data(self):
        # Download the dataset and build caches on a
        # single process first to avoid waste w/ DDP.
        self.download_and_transform()

    def setup(self, stage: str):
        # Load the dataset on each process, from cache.
        self.hf_dataset = self.download_and_transform()

    def train_dataloader(self):
        return DataLoader(
            self.hf_dataset["train"], batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(self.hf_dataset["validation"], batch_size=self.batch_size)
