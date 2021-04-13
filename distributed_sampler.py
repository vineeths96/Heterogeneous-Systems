import math
from typing import TypeVar, Optional, Iterator, List

import torch
from torch.utils.data import Sampler
from torch.utils.data.dataset import Dataset
import torch.distributed as dist
from itertools import islice


T_co = TypeVar("T_co", covariant=True)


class DistributedSampler(Sampler[T_co]):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        partitions: List,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()

        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.dataset = dataset
        self.batch_size = batch_size
        self.partition_fractions = partitions
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        self.adaptive_batch_sizes = [self.partition_fractions[worker_ind] * self.batch_size for worker_ind in range(num_replicas)]
        self.rough_partition_sizes = [math.ceil(partition * len(self.dataset)) for partition in partitions]
        self.partition_sizes = [math.ceil(partition_size / adaptive_batch_size) * math.floor(adaptive_batch_size) for partition_size, adaptive_batch_size in zip(self.rough_partition_sizes, self.adaptive_batch_sizes)]

        self.num_samples = self.partition_sizes[self.rank]
        self.total_size = sum(self.partition_sizes)

        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore
        else:
            indices = list(range(len(self.dataset)))  # type: ignore

        if len(indices) > self.total_size:
            indices = indices[: self.total_size]
        else:
            indices += indices[: self.total_size - len(indices)]

        assert len(indices) == self.total_size

        indices = [list(islice(indices, partition)) for partition in self.partition_sizes]
        indices_rank = indices[self.rank]

        assert len(indices_rank) == self.num_samples

        return iter(indices_rank)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def get_batch_size(self):
        return math.floor(self.adaptive_batch_sizes[self.rank])
