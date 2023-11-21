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
        val_split_ratio: float,
        ############################
        num_hard_prompt_tkns: int,
        hard_prompt_tkn: str,
    ):
        super().__init__()

        # logger=False since we already log hparams manually in train.py.
        self.save_hyperparameters(logger=False)

    def load_mapped_datasets(self):
        # Note: This function cannot access any properties of self directly, or it
        # will mess up deterministic serialization. Instead, pass them as arguments.
        def transform(
            x, tokenizer: Tokenizer, hard_prompt_tkn: str, num_hard_prompt_tkns: int
        ):
            seq = tokenizer.encode(
                fmt_vicuna_input(
                    f"{hard_prompt_tkn * num_hard_prompt_tkns} {x['inputs']}",
                    x["targets"],
                    space_before_prompt=False,  # We don't want a space before the hard prompt tokens.
                ),
                eos=True,  # Don't see why you wouldn't want to train with an eos_token.
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
            load_dataset("parquet", data_dir=self.hparams.data_dir, split="train").map(
                partial(
                    transform,
                    tokenizer=self.hparams.tokenizer,
                    hard_prompt_tkn=self.hparams.hard_prompt_tkn,
                    num_hard_prompt_tkns=self.hparams.num_hard_prompt_tkns,
                ),
                num_proc=32,
            )
            # After map so changing test_size doesn't bust the cache.
            # Seed so the auto shuffle is 100% idempotent, just in case.
            .train_test_split(test_size=self.hparams.val_split_ratio, seed=1337)
            # Convert all relevant types to tensors. All int32s will become int64s.
            .with_format("torch")
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
            batch_size=1,  # Can only compute hard prompt grads on 1 seq at a time at the moment.
            num_workers=8,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.hf_datasets["test"],
            collate_fn=pad_collate_fn,
            batch_size=1,  # Can only compute hard prompt grads on 1 seq at a time at the moment.
            num_workers=8,
            pin_memory=True,
        )
