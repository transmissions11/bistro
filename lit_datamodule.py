import torch

import lightning.pytorch as L

from functools import partial

from torch.utils.data import DataLoader

from datasets import load_dataset

from transformers import AutoImageProcessor

from utils.collate import collate_fn


class LitDataModule(L.LightningDataModule):
    def __init__(
        self,
        model_id: str,
        data_dir: str,
        micro_batch_size: int,
        val_split_ratio: float,
    ):
        super().__init__()

        self.processor = AutoImageProcessor.from_pretrained(model_id)

        # logger=False since we already log hparams manually in train.py.
        self.save_hyperparameters(logger=False)

    def load_mapped_datasets(self):

        # Note: This function cannot access any properties of self directly, or it
        # will mess up deterministic serialization. Instead, pass them as arguments.
        def transform(
            x,
            processor: AutoImageProcessor,
        ):
            pixel_values = processor(x["image"], return_tensors="pt").pixel_values

            labels = torch.tensor([x["lturn"], x["rturn"], x["noturn"]])

            return pixel_values.squeeze(0), labels  # Squeeze off the batch dimension.

        print("helloooooooooooooooooooooo")

        # All the data will be in the root level of data_dir,
        # so it's all considered part of the "train" split.
        return (
            load_dataset("imagefolder", data_dir=self.hparams.data_dir, split="train")
            # .map(
            #     partial(
            #         transform,
            #         tokenizer=self.hparams.tokenizer,
            #     ),
            #     num_proc=32,
            # )
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
            collate_fn=collate_fn,
            batch_size=self.hparams.micro_batch_size,
            num_workers=8,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.hf_datasets["train"],
            collate_fn=collate_fn,
            # Since we're not computing and storing gradients
            # while validating, we can use a larger batch size.
            batch_size=self.hparams.micro_batch_size * 2,
            num_workers=8,
            pin_memory=True,
            shuffle=False,
            drop_last=True,
        )
