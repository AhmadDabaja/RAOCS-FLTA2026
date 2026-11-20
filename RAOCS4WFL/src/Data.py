from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST
from torchvision.transforms import Compose, Normalize, ToTensor


# Cache the prepared federated dataset so it is built only once per run.
_DATA_MANAGER = None


def get_transforms(dataset_name: str) -> Compose:
    """
    Return the image transform associated with the selected dataset.
    """
    dataset_name = dataset_name.lower()

    if dataset_name in {"cifar10", "cifar100"}:
        return Compose(
            [
                ToTensor(),
                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    if dataset_name == "fashionmnist":
        return Compose(
            [
                ToTensor(),
                Normalize((0.5,), (0.5,)),
            ]
        )

    raise ValueError("dataset_name must be 'cifar10' or 'fashionmnist'")


def load_base_dataset(dataset_name: str, root: str = "./datasets"):
    """
    Load the raw training split of the selected dataset.

    We only use the original training split here, then perform our own
    client partitioning and local train/test splitting afterwards.
    """
    dataset_name = dataset_name.lower()
    transform = get_transforms(dataset_name)

    if dataset_name == "cifar10":
        return CIFAR10(root=root, train=True, download=True, transform=transform)

    if dataset_name == "cifar100":
        return CIFAR100(root=root, train=True, download=True, transform=transform)

    if dataset_name == "fashionmnist":
        return FashionMNIST(root=root, train=True, download=True, transform=transform)

    raise ValueError("dataset_name must be 'fashionmnist', 'cifar10', or 'cifar100'")


def extract_labels(dataset) -> np.ndarray:
    """
    Extract labels from a torchvision dataset as a NumPy array.
    """
    if not hasattr(dataset, "targets"):
        raise ValueError("The provided dataset does not expose a 'targets' attribute.")

    return np.asarray(dataset.targets)


class DirichletPartitioner:
    """
    Partition a labeled dataset into client subsets using a Dirichlet distribution.

    Each class is distributed independently across clients according to a
    Dirichlet(alpha) draw. Lower alpha produces more heterogeneous partitions.
    """

    def __init__(
        self,
        num_partitions: int,
        alpha: float,
        seed: int = 42,
        min_samples_per_partition: int = 1,
        max_retries: int = 20,
    ) -> None:
        if num_partitions <= 0:
            raise ValueError("num_partitions must be > 0")
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        if min_samples_per_partition < 0:
            raise ValueError("min_samples_per_partition must be >= 0")
        if max_retries <= 0:
            raise ValueError("max_retries must be > 0")

        self.num_partitions = num_partitions
        self.alpha = alpha
        self.seed = seed
        self.min_samples_per_partition = min_samples_per_partition
        self.max_retries = max_retries

    def partition(self, labels: np.ndarray) -> Dict[int, List[int]]:
        """
        Build client partitions from the provided label vector.

        Returns a dictionary:
            partition_id -> list of dataset indices
        """
        if len(labels) < self.num_partitions:
            raise ValueError(
                f"Cannot create {self.num_partitions} partitions from only {len(labels)} samples."
            )

        for attempt in range(self.max_retries):
            rng = np.random.default_rng(self.seed + attempt)
            partition_to_indices = self._partition_once(labels, rng)

            min_size = min(len(indices) for indices in partition_to_indices.values())
            if min_size >= self.min_samples_per_partition:
                return partition_to_indices

        raise RuntimeError(
            "Failed to create valid Dirichlet partitions. "
            "Try increasing alpha, reducing num_partitions, or lowering min_samples_per_partition."
        )

    def _partition_once(
        self,
        labels: np.ndarray,
        rng: np.random.Generator,
    ) -> Dict[int, List[int]]:
        """
        Perform one Dirichlet partitioning attempt.
        """
        partition_to_indices: Dict[int, List[int]] = {
            partition_id: [] for partition_id in range(self.num_partitions)
        }

        classes = np.unique(labels)

        for class_id in classes:
            class_indices = np.where(labels == class_id)[0]
            rng.shuffle(class_indices)

            proportions = rng.dirichlet([self.alpha] * self.num_partitions)
            raw_counts = proportions * len(class_indices)

            counts = np.floor(raw_counts).astype(int)
            remainder = len(class_indices) - int(counts.sum())

            if remainder > 0:
                fractional_parts = raw_counts - counts
                extra_indices = np.argsort(fractional_parts)[-remainder:]
                counts[extra_indices] += 1

            start = 0
            for partition_id, count in enumerate(counts):
                end = start + int(count)
                partition_to_indices[partition_id].extend(class_indices[start:end].tolist())
                start = end

        for partition_id in partition_to_indices:
            rng.shuffle(partition_to_indices[partition_id])

        return partition_to_indices


class TrainTestSplitter:
    """
    Split one client's indices into local train and test subsets.
    """

    def __init__(self, test_ratio: float, seed: int = 42) -> None:
        if not 0.0 <= test_ratio < 1.0:
            raise ValueError("test_ratio must satisfy 0.0 <= test_ratio < 1.0")

        self.test_ratio = test_ratio
        self.seed = seed

    def split(self, indices: List[int]) -> Tuple[List[int], List[int]]:
        """
        Split the provided indices into train/test subsets.

        The split is deterministic with respect to the configured seed.
        """
        if len(indices) == 0:
            return [], []

        shuffled = np.array(indices, copy=True)
        rng = np.random.default_rng(self.seed)
        rng.shuffle(shuffled)

        if self.test_ratio == 0.0:
            return shuffled.tolist(), []

        if len(shuffled) == 1:
            return shuffled.tolist(), []

        test_size = int(round(len(shuffled) * self.test_ratio))
        test_size = max(1, min(test_size, len(shuffled) - 1))

        test_indices = shuffled[:test_size].tolist()
        train_indices = shuffled[test_size:].tolist()

        return train_indices, test_indices


class FederatedDataManager:
    """
    Prepare and serve client-specific dataloaders for a federated learning simulation.
    """

    def __init__(
        self,
        dataset_name: str,
        num_partitions: int,
        alpha: float,
        test_ratio: float,
        batch_size: int,
        seed: int = 42,
        data_root: str = "./datasets",
        num_workers: int = 0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        self.dataset_name = dataset_name.lower()
        self.num_partitions = num_partitions
        self.alpha = alpha
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        self.seed = seed
        self.data_root = data_root
        self.num_workers = num_workers

        self.dataset = load_base_dataset(self.dataset_name, root=self.data_root)
        self.labels = extract_labels(self.dataset)

        self.partitioner = DirichletPartitioner(
            num_partitions=self.num_partitions,
            alpha=self.alpha,
            seed=self.seed,
            min_samples_per_partition=1,
        )
        self.splitter = TrainTestSplitter(test_ratio=self.test_ratio, seed=self.seed)

        self.partition_to_indices = self.partitioner.partition(self.labels)
        self.client_splits = self._build_client_splits()

    def _build_client_splits(self) -> Dict[int, Dict[str, List[int]]]:
        """
        Precompute the train/test indices for every client.
        """
        client_splits: Dict[int, Dict[str, List[int]]] = {}

        for partition_id, indices in self.partition_to_indices.items():
            train_indices, test_indices = self.splitter.split(indices)
            client_splits[partition_id] = {
                "train": train_indices,
                "test": test_indices,
            }

        return client_splits

    def get_client_dataloaders(self, partition_id: int) -> Tuple[DataLoader, DataLoader]:
        """
        Return the PyTorch dataloaders for one client.
        """
        if partition_id not in self.client_splits:
            raise ValueError(
                f"Invalid partition_id={partition_id}. "
                f"Valid range is [0, {self.num_partitions - 1}]."
            )

        train_indices = self.client_splits[partition_id]["train"]
        test_indices = self.client_splits[partition_id]["test"]

        train_subset = Subset(self.dataset, train_indices)
        test_subset = Subset(self.dataset, test_indices)

        train_loader = DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        test_loader = DataLoader(
            test_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        return train_loader, test_loader

    def get_client_num_examples(self, partition_id: int) -> Dict[str, int]:
        """
        Return the number of train and test samples for one client.
        """
        if partition_id not in self.client_splits:
            raise ValueError(
                f"Invalid partition_id={partition_id}. "
                f"Valid range is [0, {self.num_partitions - 1}]."
            )

        return {
            "train": len(self.client_splits[partition_id]["train"]),
            "test": len(self.client_splits[partition_id]["test"]),
        }


def prepare_federated_data(
    dataset_name: str,
    num_partitions: int,
    alpha: float,
    test_ratio: float,
    batch_size: int,
    seed: int = 42,
    data_root: str = "./datasets",
    num_workers: int = 0,
) -> FederatedDataManager:
    """
    Prepare the federated dataset once and cache it globally.
    """
    global _DATA_MANAGER

    _DATA_MANAGER = FederatedDataManager(
        dataset_name=dataset_name,
        num_partitions=num_partitions,
        alpha=alpha,
        test_ratio=test_ratio,
        batch_size=batch_size,
        seed=seed,
        data_root=data_root,
        num_workers=num_workers,
    )
    return _DATA_MANAGER


def load_data(
    dataset_name: str,
    partition_id: int,
    num_partitions: int,
    alpha: float,
    test_ratio: float,
    batch_size: int,
    seed: int = 42,
    data_root: str = "./datasets",
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Return the train and test dataloaders for one client.

    If the federated dataset has not been prepared yet, it is created automatically.
    """
    global _DATA_MANAGER

    if _DATA_MANAGER is None:
        prepare_federated_data(
            dataset_name=dataset_name,
            num_partitions=num_partitions,
            alpha=alpha,
            test_ratio=test_ratio,
            batch_size=batch_size,
            seed=seed,
            data_root=data_root,
            num_workers=num_workers,
        )

    return _DATA_MANAGER.get_client_dataloaders(partition_id)

