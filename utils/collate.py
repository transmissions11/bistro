import torch


def collate_fn(batch):
    data = torch.stack([item["pixel_values"] for item in batch])  # pixel values
    target = torch.stack([item["labels"] for item in batch])  # labels
    return data, target
