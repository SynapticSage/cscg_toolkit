"""
Vision dataset loaders (MNIST, CIFAR, etc.)

Created: 2025-11-09
"""

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


def get_mnist(batch_size=128, data_dir='./data', num_workers=4):
    """Load standard MNIST dataset.

    Args:
        batch_size: Batch size for training/testing
        data_dir: Directory to store/load data
        num_workers: Number of data loading workers

    Returns:
        train_loader: Training DataLoader
        test_loader: Test DataLoader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean/std
    ])

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader


def get_sequential_mnist(batch_size=128, data_dir='./data', num_workers=4):
    """Load MNIST for sequential processing (pixel-by-pixel).

    Each image is flattened to 784-step sequence.

    Args:
        batch_size: Batch size for training/testing
        data_dir: Directory to store/load data
        num_workers: Number of data loading workers

    Returns:
        train_loader: Training DataLoader
        test_loader: Test DataLoader
    """
    # No normalization for sequential - keep pixels in [0, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    # Wrap with sequential transform
    train_dataset = SequentialMNISTDataset(train_dataset)
    test_dataset = SequentialMNISTDataset(test_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader


class SequentialMNISTDataset(Dataset):
    """Wrapper that flattens MNIST images to sequences.

    Converts (1, 28, 28) images to (784, 1) sequences.
    """

    def __init__(self, mnist_dataset):
        self.mnist = mnist_dataset

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        image, label = self.mnist[idx]

        # Flatten image to sequence
        sequence = image.view(-1, 1)  # (784, 1)

        return sequence, label


def get_cifar10(batch_size=128, data_dir='./data', num_workers=4):
    """Load CIFAR-10 dataset.

    Args:
        batch_size: Batch size for training/testing
        data_dir: Directory to store/load data
        num_workers: Number of data loading workers

    Returns:
        train_loader: Training DataLoader
        test_loader: Test DataLoader
    """
    # Standard CIFAR-10 augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform_train
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform_test
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader
