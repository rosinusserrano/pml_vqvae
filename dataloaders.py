"Module for getting datasets/dataloaders"
from torch.utils.data import DataLoader

from torchvision.datasets.cifar import CIFAR10
from torchvision.datasets.imagenet import ImageNet
from torchaudio.datasets.vctk import VCTK_092
from torchvision.transforms import ToTensor


def load_imagenet(root: str, batch_size: int = 128):
    """Loads the ImageNet dataset.

    `root` is the path to the already downloaded ImageNet
    dataset.
    
    Returns a tuple of 2 pytorch dataloaders. First is
    the train dataset and second is the test dataset. 
    """
    train_set = ImageNet(root=root, split="train")
    test_set = ImageNet(root=root, split="test")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    return train_loader, test_loader


def load_cifar10(root: str | None, batch_size: int = 128):
    """Loads the CIFAR10 dataset
    
    `root` is the path to the dataset. If `root` is empty,
    the dataset will be downloaded to that directory.
    
    Returns a tuple of 2 pytorch dataloaders. First is
    the train dataset and second is the test dataset."""
    train_set = CIFAR10(root=root, train=True, download=True, transform=ToTensor())
    test_set = CIFAR10(root=root, train=False, download=True, transform=ToTensor())

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    return train_loader, test_loader


def load_vctk(root: str, batch_size: int = 128):
    """Loads the VCTK 0.92 dataset
    
    `root` is the path to the dataset. If `root` is empty,
    the dataset will be downloaded to that directory.
    
    Returns a pytorch dataloader for the dataset."""
    dataset = VCTK_092(root=root, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader