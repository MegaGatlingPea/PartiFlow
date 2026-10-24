import math
from collections import Counter
from typing import Optional, Sequence

import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler


class TaggedRatioDistributedSampler(DistributedSampler):
    """Sample a tagged dataset with an epoch-level anchor/fill ratio.

    The sampler builds one deterministic global index list per epoch, pads or drops it
    to make it evenly divisible across ranks, then returns the rank-local strided
    slice. It subclasses ``DistributedSampler`` so Lightning will not wrap it again.
    """

    def __init__(
        self,
        dataset: Dataset,
        tags: Sequence[str],
        anchor_tag: str,
        fill_tag: str,
        anchor_ratio: float,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank() if dist.is_initialized() else 0
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )
        if len(tags) != len(dataset):
            raise ValueError(f"tags length {len(tags)} must match dataset length {len(dataset)}")
        if not 0 < anchor_ratio <= 1:
            raise ValueError(f"anchor_ratio must be in (0, 1], got {anchor_ratio}")

        self.dataset = dataset
        self.tags = [str(tag) for tag in tags]
        self.anchor_tag = str(anchor_tag)
        self.fill_tag = str(fill_tag)
        self.anchor_ratio = float(anchor_ratio)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        self.anchor_indices = [idx for idx, tag in enumerate(self.tags) if tag == self.anchor_tag]
        self.fill_indices = [idx for idx, tag in enumerate(self.tags) if tag == self.fill_tag]
        if not self.anchor_indices:
            counts = Counter(self.tags)
            raise ValueError(f"No samples found for anchor_tag={self.anchor_tag!r}; tag counts={dict(counts)}")
        if self.anchor_ratio < 1.0 and not self.fill_indices:
            counts = Counter(self.tags)
            raise ValueError(f"No samples found for fill_tag={self.fill_tag!r}; tag counts={dict(counts)}")

        desired_global_size = max(
            len(self.anchor_indices),
            math.ceil(len(self.anchor_indices) / self.anchor_ratio),
        )
        if self.drop_last and desired_global_size % self.num_replicas != 0:
            self.num_samples = math.ceil((desired_global_size - self.num_replicas) / self.num_replicas)
        else:
            self.num_samples = math.ceil(desired_global_size / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.global_epoch_size = self.total_size
        self.num_fill_samples = self.global_epoch_size - len(self.anchor_indices)
        if self.num_fill_samples > 0 and self.anchor_ratio < 1.0 and not self.fill_indices:
            counts = Counter(self.tags)
            raise ValueError(f"No samples found for fill_tag={self.fill_tag!r}; tag counts={dict(counts)}")

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        anchor = torch.tensor(self.anchor_indices, dtype=torch.long)
        if self.shuffle:
            anchor = anchor[torch.randperm(anchor.numel(), generator=g)]
        indices = anchor.tolist()

        if self.num_fill_samples > 0:
            fill_source = self.fill_indices if self.fill_indices else self.anchor_indices
            fill = torch.tensor(fill_source, dtype=torch.long)
            fill_pick = fill[torch.randint(fill.numel(), (self.num_fill_samples,), generator=g)]
            indices.extend(fill_pick.tolist())

        if self.shuffle:
            order = torch.randperm(len(indices), generator=g).tolist()
            indices = [indices[idx] for idx in order]

        assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
