import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
import numpy as np
import os

def get_transform():
    """Returns basic transformation for CIFAR-10 dataset"""
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    return transform

def load_cifar10_data():
    """Loads CIFAR-10 dataset with given transformation"""
    transform = get_transform()
    train_dataset = torchvision.datasets.CIFAR10(root='./data/raw', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data/raw', train=False, download=True, transform=transform)
    return train_dataset, test_dataset

def select_sample(dataset, num_samples):
    """Selects num_samples from each class in the dataset"""
    targets = np.array(dataset.targets)
    indices = []
    for cls in range(10):
        cls_indices = np.where(targets == cls)[0][:num_samples]
        indices.extend(cls_indices)
    return Subset(dataset, indices)

def save_dataset(dataset, filename):
    """Saves the dataset to a file"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(dataset, filename)