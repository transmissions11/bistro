import torch
from torch.utils.data import DataLoader

from functools import partial

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
        print("download_and_transform!!")

        # Note: This function cannot access any properties of self directly, or it
        # will mess up deterministic serialization. Instead, pass them as arguments.
        def transform(
            x, tokenizer: Tokenizer, soft_prompt_tkn: str, num_soft_prompt_tkns: int
        ):
            seq = tokenizer.encode(
                fmt_vicuna_input(
                    f"{soft_prompt_tkn * num_soft_prompt_tkns} {x['prompt']}",
                    x["response"],
                )
            )

            return {
                "input_ids": seq[:-1],
                # Mask everything before the assistant response.
                "targets": mask_before_inclusive(
                    VICUNA_END_OF_USER_PROMPT_SEQUENCE, seq[1:], tokenizer
                ),
            }

        return (
            load_dataset("parquet", data_dir=self.data_dir)
            .map(
                partial(
                    transform,
                    tokenizer=self.tokenizer,
                    soft_prompt_tkn=self.soft_prompt_tkn,
                    num_soft_prompt_tkns=self.num_soft_prompt_tkns,
                ),
                remove_columns=["prompt", "response"],
                num_proc=32,
            )
            .with_format("torch")
        )

    def prepare_data(self):
        print("prepare!")
        # Download the dataset and build caches on a
        # single process first to avoid waste w/ DDP.
        self.download_and_transform()

    def setup(self, stage: str):
        print("setup!")
        # Load the dataset on each process, from cache.
        self.hf_dataset = self.download_and_transform()

    def train_dataloader(self):
        # TODO: try collate
        return DataLoader(
            self.hf_dataset["train"], batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(self.hf_dataset["validation"], batch_size=self.batch_size)
