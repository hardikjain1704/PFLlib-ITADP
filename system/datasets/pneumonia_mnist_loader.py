"""
PneumoniaMNIST Dataset Loader.

Downloads PneumoniaMNIST via the medmnist package and shards it for
federated learning across two laptops.

Usage:
    from datasets.pneumonia_mnist_loader import load_pneumonia_shard

    train_loader, test_loader, info = load_pneumonia_shard(
        shard_id=0,        # 0 for Laptop A, 1 for Laptop B
        num_shards=2,
        batch_size=32,
    )
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import medmnist
from medmnist import PneumoniaMNIST


def load_pneumonia_shard(
    shard_id: int = 0,
    num_shards: int = 2,
    batch_size: int = 32,
    data_root: str = None,
):
    """Load one shard of PneumoniaMNIST for federated training.

    Args:
        shard_id:   Which shard (0 … num_shards-1).
        num_shards: Total number of clients/shards.
        batch_size: DataLoader batch size.
        data_root:  Where to cache the dataset download.

    Returns:
        (train_loader, test_loader, info_dict)
    """
    if data_root is None:
        data_root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..", "dataset", "PneumoniaMNIST",
        )
    os.makedirs(data_root, exist_ok=True)

    # Download / load full datasets
    train_dataset = PneumoniaMNIST(
        split="train", download=True, root=data_root, transform=None,
    )
    test_dataset = PneumoniaMNIST(
        split="test", download=True, root=data_root, transform=None,
    )

    # Shard training data evenly
    n_train = len(train_dataset)
    indices = list(range(n_train))
    np.random.seed(42)  # deterministic sharding
    np.random.shuffle(indices)
    shard_size = n_train // num_shards
    start = shard_id * shard_size
    end = start + shard_size if shard_id < num_shards - 1 else n_train
    shard_indices = indices[start:end]

    train_subset = _PneumoniaSubset(train_dataset, shard_indices)
    test_wrapper = _PneumoniaSubset(test_dataset, list(range(len(test_dataset))))

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_wrapper, batch_size=batch_size, shuffle=False)

    info = {
        "shard_id": shard_id,
        "num_shards": num_shards,
        "train_samples": len(shard_indices),
        "test_samples": len(test_dataset),
        "num_classes": 2,
        "image_size": 28,
        "channels": 1,
        "total_train": n_train,
    }

    return train_loader, test_loader, info


class _PneumoniaSubset(torch.utils.data.Dataset):
    """Wraps medmnist dataset subset and converts to float tensor."""

    def __init__(self, base_dataset, indices):
        self.base = base_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.base[self.indices[idx]]
        # medmnist returns PIL Image; convert to tensor
        if not isinstance(img, torch.Tensor):
            import torchvision.transforms as T
            img = T.ToTensor()(img)  # -> (1, 28, 28) float [0,1]
        label = int(label.squeeze())
        return img, label


def get_dataset_features():
    """Return declared features of this dataset (for purpose validation)."""
    return ["image", "label"]


if __name__ == "__main__":
    # Quick test
    for shard in range(2):
        train_ld, test_ld, info = load_pneumonia_shard(shard_id=shard, num_shards=2)
        print(f"Shard {shard}: {info}")
        x, y = next(iter(train_ld))
        print(f"  Batch shape: {x.shape}, labels: {y[:8].tolist()}")
