import torch

import lightning.pytorch as L

from torch.utils.data import DataLoader, Dataset, random_split

from transformers import AutoImageProcessor

from utils.collate import collate_fn

from PIL import Image

import numpy as np

import pandas as pd

import os


class MultiLabelDataset(Dataset):
    def __init__(self, data_dir, df, processor, num_ctx_frames):
        self.data_dir = data_dir
        self.df = df
        self.processor = processor
        self.num_ctx_frames = num_ctx_frames

    def load_and_process_image(self, frame_item):
        image = Image.open(
            os.path.join(self.data_dir, frame_item["file_name"])
        ).convert("RGB")
        return self.processor(
            image, return_tensors="pt"
        ).pixel_values  # [1, C=3, hidden_size, hidden_size]

    def __getitem__(self, idx):
        item = self.df.iloc[idx]

        if idx <= self.num_ctx_frames:
            pixel_values = self.load_and_process_image(self.df.iloc[idx])
            print("pixel_values.shape", pixel_values.shape)
            frames = torch.cat(
                [pixel_values] * self.num_ctx_frames, dim=0
            )  # [num_ctx_frames, C=3, hidden_size, hidden_size]
            print("frames.shape", frames.shape)
        else:
            frames = torch.cat(
                [
                    self.load_and_process_image(self.df.iloc[i])
                    for i in range(idx - (self.num_ctx_frames - 1), idx + 1)
                ],
                dim=0,
            )  # [num_ctx_frames, C=3, hidden_size, hidden_size]

        labels = torch.from_numpy(item[1:].values.astype(np.int32))  # [num_classes]

        # print("frames.shape", frames.shape, "labels.shape", labels.shape)

        return frames, labels

    def __len__(self):
        return len(self.df)


class LitDataModule(L.LightningDataModule):
    def __init__(
        self,
        model_id: str,
        data_dir: str,
        num_ctx_frames: int,
        micro_batch_size: int,
        val_split_ratio: float,
    ):
        super().__init__()

        self.processor = AutoImageProcessor.from_pretrained(model_id)

        self.df = pd.read_csv(os.path.join(data_dir, "metadata.csv"))

        # logger=False since we already log hparams manually in train.py.
        self.save_hyperparameters(logger=False)

    def setup(self, stage: str):
        dataset = MultiLabelDataset(
            data_dir=self.hparams.data_dir,
            df=self.df,
            processor=self.processor,
            num_ctx_frames=self.hparams.num_ctx_frames,
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
            num_workers=8,
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
            num_workers=8,
            pin_memory=True,
            shuffle=False,
            drop_last=True,
        )
