import lightning.pytorch as L

from torch.utils.data import DataLoader, Dataset

from PIL import Image

import numpy as np
import torch

from datasets import load_dataset

from torchvision.transforms import Compose, Resize, ToTensor, Normalize

import pandas as pd

from transformers import AutoImageProcessor

import os


class MultiLabelDataset(Dataset):
    def __init__(self, root, df, transform):
        self.root = root
        self.df = df
        self.transform = transform

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        # get image
        image_path = os.path.join(self.root, item["file_path"])
        image = Image.open(image_path).convert("RGB")

        # prepare image for the model
        pixel_values = self.transform(image)

        # get labels
        labels = item[1:].values.astype(
            np.float32
        )  # TODO: verify this eventually ends up as bfloat16

        # turn into PyTorch tensor
        labels = torch.from_numpy(labels)

        return pixel_values, labels

    def __len__(self):
        return len(self.df)


def collate_fn(batch):
    data = torch.stack([item[0] for item in batch])  # pixel values
    target = torch.stack([item[1] for item in batch])  # class values vector
    return data, target


class LitDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        micro_batch_size: int,
        val_split_ratio: float,
    ):
        super().__init__()

        # logger=False since we already log hparams manually in train.py.
        self.save_hyperparameters(logger=False)

    def load_mapped_datasets(self):

        df = pd.read_csv("FINAL_expanded_2_3.csv")
        df.head()

        model_id = "google/siglip-so400m-patch14-384"
        processor = AutoImageProcessor.from_pretrained(model_id)

        # get appropriate size, mean and std based on the image processor
        size = processor.size["height"]
        mean = processor.image_mean
        std = processor.image_std

        transform = Compose(
            [
                Resize((size, size)),
                ToTensor(),
                Normalize(mean=mean, std=std),
            ]
        )

        return {"train": MultiLabelDataset(root="./", df=df, transform=transform)}

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

    # def val_dataloader(self):
    #     return DataLoader(
    #         self.hf_datasets["test"],
    #         collate_fn=pad_collate_fn,
    #         # Since we're not computing and storing gradients
    #         # while validating, we can use a larger batch size.
    #         batch_size=self.hparams.micro_batch_size * 2,
    #         num_workers=8,
    #         pin_memory=True,
    #         shuffle=False,
    #         drop_last=True,
    #     )
