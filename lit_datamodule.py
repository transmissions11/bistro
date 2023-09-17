import torch.multiprocessing as mp
import multiprocessing as mp2
from multiprocess import set_start_method

set_start_method("spawn", force=True)

mp.set_start_method("spawn", force=True)
mp2.set_start_method("spawn", force=True)

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

    def download_and_transform(soft_prompt_tkn, num_soft_prompt_tkns, data_dir):
        def transform(x):
            seq = fmt_vicuna_input(
                f"{soft_prompt_tkn * num_soft_prompt_tkns} {x['prompt']}",
                x["response"],
            )

            return {
                "input_ids": seq[:-1],
                # Mask everything before the assistant response.
                "targets": seq[1:],
            }

        return (
            load_dataset("parquet", data_dir=data_dir)
            .map(
                transform,
                remove_columns=["prompt", "response"],
                load_from_cache_file=False,  # TODO: Fix this.
                num_proc=8,
                # We can force cache like this if needed.
                # new_fingerprint="t1"
            )
            .with_format("torch")
        )

    def prepare_data(self):
        # Download the dataset and build caches on a
        # single process first to avoid waste w/ DDP.
        self.download_and_transform(self.soft_prompt_tkn, self.num_soft_prompt_tkns)

    def setup(self, stage: str):
        # Load the dataset on each process, from cache.
        self.hf_dataset = self.download_and_transform(
            self.soft_prompt_tkn, self.num_soft_prompt_tkns, self.data_dir
        )

    def train_dataloader(self):
        # TODO: try collate
        return DataLoader(
            self.hf_dataset["train"], batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(self.hf_dataset["validation"], batch_size=self.batch_size)
