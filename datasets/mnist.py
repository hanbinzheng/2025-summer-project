from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Tuple, Optional

def get_mnist_dataloader(
        root: str = "data/data_mnist",
        batch_size: int = 128,
        num_workers: int = 4,
        train: bool = True,
        test: bool = False,
        download: bool = True
) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
    train_loader = None
    test_loader = None
    mean = (0.5, )
    std = (0.5, )

    if train:
        train_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        train_dataset = datasets.MNIST(
            root = root,
            train = True,
            download = download,
            transform = train_transform
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size = batch_size,
            shuffle = True,
            num_workers = num_workers
        )

    if test:
        test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        test_dataset = datasets.MNIST(
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
