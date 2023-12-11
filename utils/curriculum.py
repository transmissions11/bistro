from utils.padding import pad_collate_fn


class CurriculumCollate:
    def __init__(self):
        self.prev_samples = []

    def __call__(self, batch):
        self.prev_samples.extend(batch)

        import ipdb

        ipdb.set_trace(
            cond=(0 == torch.distributed.get_rank())
            if torch.distributed.is_initialized()
            else True
        )

        return pad_collate_fn(batch)
