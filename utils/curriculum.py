from utils.padding import pad_collate_fn


class CurriculumCollate:
    def __init__(self):
        self.prev_samples = []

    def __call__(self, batch):
        self.prev_samples.extend(batch)

        return pad_collate_fn(batch)
