import os
from torchvision import datasets, transforms
import torch


def load_dataset(dataset='MNIST', batch_size=100, dataset_path='../../data', is_cuda=False, num_workers=8):
    kwargs = {'num_workers': num_workers, 'pin_memory': True} if is_cuda else {}
    if dataset == 'MNIST':
        num_classes = 10
        dataset_train = datasets.MNIST(os.path.join(dataset_path, 'MNIST'), train=True, download=False,
                                       transform=transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(os.path.join(dataset_path, 'MNIST'), train=False, transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=False, **kwargs)

    elif dataset == 'FashionMNIST':
        num_classes = 10
        dataset_train = datasets.FashionMNIST(os.path.join(dataset_path, 'FashionMNIST'), train=True, download=False,
                                              transform=transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(os.path.join(dataset_path, 'FashionMNIST'), train=False,
                                  transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=False, **kwargs)

    elif dataset == 'CIFAR10':
        num_classes = 10
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.424, 0.415, 0.384), (0.283, 0.278, 0.284))
        ])
        dataset_train = datasets.CIFAR10(os.path.join(dataset_path, 'CIFAR10'), train=True, download=False,
                                         transform=train_transform)
        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(os.path.join(dataset_path, 'CIFAR10'), train=False,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.424, 0.415, 0.384), (0.283, 0.278, 0.284))
                             ])),
            batch_size=batch_size, shuffle=False, **kwargs)

    elif dataset == 'SVHN':
        num_classes = 10
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.431, 0.430, 0.446), (0.197, 0.198, 0.199))
        ])
        dataset_train = torch.utils.data.ConcatDataset((
            datasets.SVHN(os.path.join(dataset_path, 'SVHN'), split='train', download=False, transform=train_transform),
            # datasets.SVHN('../data/SVHN', split='extra', download=True, transform=train_transform))
        ))
        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.SVHN(os.path.join(dataset_path, 'SVHN'), split='test', download=False,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.431, 0.430, 0.446), (0.197, 0.198, 0.199))
                          ])),
            batch_size=batch_size, shuffle=False, **kwargs)
    else:
        raise Exception('No valid dataset is specified.')
    return train_loader, test_loader, num_classes
