from utils.padding import pad_collate_fn

import torch


class CurriculumCollate:
    def __init__(self):
        self.prev_samples = []
        self.num_learned_samples = 0

    def __call__(self, batch):
        self.prev_samples.extend(batch)

        return pad_collate_fn(self.prev_samples[: (self.num_learned_samples + 1)])
