import torch


def collate_fn(batch):
    import ipdb

    ipdb.set_trace(
        cond=(
            (0 == torch.distributed.get_rank())
            if torch.distributed.is_initialized()
            else True
        )
    )

    data = torch.stack([item[0] for item in batch])  # pixel values
    target = torch.stack([item[1] for item in batch])  # labels
    return data, target
