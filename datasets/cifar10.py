from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Tuple, Optional

def get_cifar10_dataloader(
        root: str = "data/data_cifar10",
        batch_size: int = 128,
        num_workers: int = 4,
        train: bool = True,
        test: bool = True,
        download: bool = True
) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
    # The function to get the dataloader
    train_loader = None
    test_loader = None
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    if train:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        train_dataset = datasets.CIFAR10(
            root = root,
            train = True,
            download = download,
            transform = train_transform
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size = batch_size,
            shuffle = True,
            num_workers = num_workers,
        )

    if test:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        test_dataset = datasets.CIFAR10(
            root = root,
            train = False,
            download = download,
            transform = test_transform
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size = batch_size,
            shuffle = False,
            num_workers = num_workers
        )

    return (train_loader, test_loader)
