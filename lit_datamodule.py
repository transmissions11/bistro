import lightning.pytorch as L

from functools import partial

from torch.utils.data import DataLoader

from lit_gpt import Tokenizer

from datasets import load_dataset

from utils.padding import pad_collate_fn
from utils.masking import mask_before_inclusive
from utils.vicuna import VICUNA_END_OF_USER_PROMPT_SEQUENCE, fmt_vicuna_input


class LitDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        tokenizer: Tokenizer,
        micro_batch_size: int,
        val_split_ratio: float,
        num_soft_prompt_tkns: int,
        soft_prompt_tkn: str,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.micro_batch_size = micro_batch_size
        self.val_split_ratio = val_split_ratio
        self.num_soft_prompt_tkns = num_soft_prompt_tkns
        self.soft_prompt_tkn = soft_prompt_tkn

    def load_mapped_datasets(self):
        # Note: This function cannot access any properties of self directly, or it
        # will mess up deterministic serialization. Instead, pass them as arguments.
        def transform(
            x, tokenizer: Tokenizer, soft_prompt_tkn: str, num_soft_prompt_tkns: int
        ):
            seq = tokenizer.encode(
                fmt_vicuna_input(
                    f"{soft_prompt_tkn * num_soft_prompt_tkns} {x['inputs']}",
                    x["targets"],
                )
            )

            return {
                "inputs": seq[:-1],
                # Mask everything before the assistant response.
                "targets": mask_before_inclusive(
                    VICUNA_END_OF_USER_PROMPT_SEQUENCE, seq[1:], tokenizer
                ),
            }

        return (
            # All the data will be in the root level of data_dir,
            # so it's all considered part of the "train" split.
            load_dataset("parquet", data_dir=self.data_dir, split="train")
            .map(
                partial(
                    transform,
                    tokenizer=self.tokenizer,
                    soft_prompt_tkn=self.soft_prompt_tkn,
                    num_soft_prompt_tkns=self.num_soft_prompt_tkns,
                ),
                num_proc=32,
            )
            # After map so changing test_size doesn't bust the cache.
            # Seed so the auto shuffle is 100% idempotent, just in case.
            .train_test_split(test_size=self.val_split_ratio, seed=1337)
            .with_format("torch")  # Convert relevant types to tensors.
        )

    def prepare_data(self):
        # Download the dataset and build caches on a
        # single process first to avoid waste w/ DDP.
        self.load_mapped_datasets()

    def setup(self, stage: str):
        # Load the dataset on each process, from cache.
        self.hf_datasets = self.load_mapped_datasets()

    def train_dataloader(self):
        return DataLoader(
            self.hf_datasets["train"],
            collate_fn=pad_collate_fn,
            batch_size=self.micro_batch_size,
            num_workers=8,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        # TODO: double batch size?
        return DataLoader(
            self.hf_datasets["test"],
            collate_fn=pad_collate_fn,
            # Since we're not computing and storing gradients
            # while validating, we can use a larger batch size.
            batch_size=self.micro_batch_size * 2,
            num_workers=8,
            pin_memory=True,
        )
