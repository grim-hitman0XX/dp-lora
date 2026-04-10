"""Virtual batch manager for simulating large logical batches with limited memory."""

from __future__ import annotations

import math
from typing import List

import numpy as np
from torch.utils.data import DataLoader, Sampler

from opacus.utils.uniform_sampler import (
    DistributedUniformWithReplacementSampler,
    UniformWithReplacementSampler,
)


class BatchSplittingSampler(Sampler[List[int]]):
    """Wraps a batch sampler, splitting large batches into smaller chunks.

    For each logical batch from the underlying sampler:
    - Splits into ceil(len(batch) / max_batch_size) micro-batches
    - Signals the optimizer to skip on all but the last micro-batch

    This ensures the training loop can call optimizer.step() on every
    micro-batch, but the optimizer only performs the actual update
    (noise + step) on the last one.
    """

    def __init__(
        self,
        *,
        sampler: Sampler[List[int]],
        max_batch_size: int,
        optimizer,
    ):
        self.sampler = sampler
        self.max_batch_size = max_batch_size
        self.optimizer = optimizer

    def __iter__(self):
        for batch_idxs in self.sampler:
            if len(batch_idxs) == 0:
                self.optimizer.signal_skip_step(do_skip=False)
                yield []
                continue

            split_idxs = np.array_split(
                batch_idxs, math.ceil(len(batch_idxs) / self.max_batch_size)
            )
            split_idxs = [s.tolist() for s in split_idxs]

            # Signal skip for all micro-batches except the last
            for x in split_idxs[:-1]:
                self.optimizer.signal_skip_step(do_skip=True)
                yield x

            # Last micro-batch: don't skip
            self.optimizer.signal_skip_step(do_skip=False)
            yield split_idxs[-1]

    def __len__(self):
        if isinstance(self.sampler, UniformWithReplacementSampler) or isinstance(
            self.sampler, DistributedUniformWithReplacementSampler
        ):
            expected_batch_size = self.sampler.sample_rate * self.sampler.num_samples
            return math.ceil(
                len(self.sampler) * (expected_batch_size / self.max_batch_size)
            )
        return len(self.sampler)


class VirtualBatchManager:
    """Context manager that wraps a DataLoader with batch splitting.

    Usage::

        with VirtualBatchManager(
            data_loader=dp_train_loader,
            max_physical_batch_size=32,
            optimizer=dp_optimizer,
        ) as vb_loader:
            for batch in vb_loader:
                optimizer.zero_grad()
                loss = model(**batch).loss
                loss.backward()
                grads = engine.grad_sample_module.get_per_sample_grads()
                optimizer.step(grads)
                engine.grad_sample_module.clear_per_sample_grads()

    The training loop is identical to non-virtual-batching code. The optimizer
    internally defers noise + update until the last micro-batch of each logical
    batch.

    Args:
        data_loader: A DataLoader (typically Poisson-sampled via DPDataLoader).
        max_physical_batch_size: Maximum number of samples per micro-batch.
        optimizer: DPOptimizer instance to coordinate skip signals with.
    """

    def __init__(
        self,
        *,
        data_loader: DataLoader,
        max_physical_batch_size: int,
        optimizer,
    ):
        self.data_loader = data_loader
        self.max_physical_batch_size = max_physical_batch_size
        self.optimizer = optimizer

    def __enter__(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_loader.dataset,
            batch_sampler=BatchSplittingSampler(
                sampler=self.data_loader.batch_sampler,
                max_batch_size=self.max_physical_batch_size,
                optimizer=self.optimizer,
            ),
            collate_fn=self.data_loader.collate_fn,
            num_workers=self.data_loader.num_workers,
            pin_memory=self.data_loader.pin_memory,
        )

    def __exit__(self, *args):
        pass
