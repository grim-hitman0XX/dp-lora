"""Poisson-sampled DataLoader using Opacus's DPDataLoader."""

from torch.utils.data import DataLoader

from opacus.data_loader import DPDataLoader


def create_poisson_dataloader(
    data_loader: DataLoader, *, distributed: bool = False
) -> DPDataLoader:
    """Wrap an existing DataLoader with Poisson batch sampling.

    Each record is included in a batch independently with probability
    q = batch_size / dataset_size. This is required for DP-SGD's
    privacy amplification by subsampling guarantee.

    Args:
        data_loader: A standard PyTorch DataLoader to wrap.
        distributed: Whether to use distributed Poisson sampling (for DDP).

    Returns:
        A DPDataLoader with Poisson batch sampling.
    """
    return DPDataLoader.from_data_loader(data_loader, distributed=distributed)
