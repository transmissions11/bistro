import torch

import lightning.pytorch as L

from torch.utils.data import DataLoader, Dataset, random_split

from transformers import AutoImageProcessor

from utils.collate import collate_fn

from PIL import Image

import polars as pl

import os


class MultiLabelDataset(Dataset):
    def __init__(self, data_dir, df, processor):
        self.data_dir = data_dir
        self.df = df
        self.processor = processor

    def __getitem__(self, idx):
        file_name, *labels = self.df.row(idx)

        image = Image.open(os.path.join(self.data_dir, file_name)).convert("RGB")

        pixel_values = self.processor(image, return_tensors="pt").pixel_values

        labels = torch.tensor(labels).nonzero(as_tuple=True)[0]  # [1]

        return pixel_values.squeeze(0), labels  # Squeeze off the batch dimension.

    def __len__(self):
        return len(self.df)


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

        self.df = pl.read_csv(os.path.join(data_dir, "metadata.csv"))

        # logger=False since we already log hparams manually in train.py.
        self.save_hyperparameters(logger=False)

    def setup(self, stage: str):
        dataset = MultiLabelDataset(
            data_dir=self.hparams.data_dir,
            df=self.df,
            processor=self.processor,
        )

        test_size = int(len(dataset) * self.hparams.val_split_ratio)
        train_size = len(dataset) - test_size

        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        if self.trainer.is_global_zero:
            print(
                f"Training on {train_size:,} examples and testing on {test_size:,} examples."
            )

        self.datasets = {"train": train_dataset, "test": test_dataset}

    def train_dataloader(self):
        return DataLoader(
            self.datasets["train"],
            collate_fn=collate_fn,
            batch_size=self.hparams.micro_batch_size,
            num_workers=16,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets["test"],
            collate_fn=collate_fn,
            # Since we're not computing and storing gradients
            # while validating, we can use a larger batch size.
            batch_size=self.hparams.micro_batch_size * 4,
            num_workers=16,
            pin_memory=True,
            shuffle=False,
            drop_last=True,
            persistent_workers=True,
        )
