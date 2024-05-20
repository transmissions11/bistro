import torch


def collate_fn(batch):
    print(batch)

    data = torch.stack([item[0] for item in batch])  # pixel values
    target = torch.stack([item[1] for item in batch])  # labels
    return data, target
