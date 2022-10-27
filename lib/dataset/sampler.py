from typing import Optional, Iterator
from collections import Counter

import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

class DistributedTarSampler(DistributedSampler):
    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        self.seed = seed
        self.drop_last = drop_last
        super(DistributedTarSampler, self).__init__(dataset, num_replicas, rank, shuffle)#, seed, drop_last

        # Split to nearest available length that is evenly divisible.
        # This is to ensure each rank receives the same amount of data when
        # using this Sampler.
        self.num_samples = num_samples
        self.total_size = min(self.num_samples) * self.num_replicas

    def __iter__(self) -> Iterator:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.num_samples[self.rank], generator=g).tolist()  # type: ignore
        else:
            indices = list(range(self.num_samples[self.rank]))  # type: ignore
        indices = indices[:min(self.num_samples)]
        assert len(indices) == min(self.num_samples)
        return iter(indices)

    def __len__(self) -> int:
        return min(self.num_samples)
