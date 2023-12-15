import os
import math
from torchvision import datasets, transforms
import torch
import numpy as np
import h5py

def getData(dataset):
    dataset = dataset
    root_path = '/home/yangqu/data/'+dataset+'/'
    train_file = h5py.File(os.path.join(root_path, dataset.lower()+'_train.h5'), 'r')
    test_file = h5py.File(os.path.join(root_path, dataset.lower()+'_test.h5'), 'r')

    x_train = train_file['spikes']
    y_train = train_file['labels']
    x_test = test_file['spikes']
    y_test = test_file['labels']
    return (x_train, y_train), (x_test, y_test)

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


def generate_copying_sequence(T, labels, c_length):
    items = [1, 2, 3, 4, 5, 6, 7, 8, 0, 9]
    x = []
    y = []

    ind = np.random.randint(labels, size=c_length)
    for i in range(c_length):
        x.append([items[ind[i]]])
    for i in range(T - 1):
        x.append([items[8]])
    x.append([items[9]])
    for i in range(c_length):
        x.append([items[8]])

    for i in range(T + c_length):
        y.append([items[8]])
    for i in range(c_length):
        y.append([items[ind[i]]])

    x = np.array(x)
    y = np.array(y)
    xx = np.array([x])
    yy = np.array([y])

    return torch.FloatTensor(xx), torch.LongTensor(yy)


def create_dataset(size, T, c_length=10, class_n=8):
    d_x = []
    d_y = []
    for i in range(size):
        sq_x, sq_y = generate_copying_sequence(T, class_n, c_length)
        sq_x, sq_y = sq_x[0], sq_y[0]
        d_x.append(sq_x)
        d_y.append(sq_y)

    d_x = torch.stack(d_x)
    d_y = torch.stack(d_y)
    return d_x, d_y


def adding_problem_generator(N, seq_len=10):
    X_num = torch.rand((N, seq_len))
    X_mask = torch.zeros((N, seq_len))
    Y = torch.zeros(N)
    for i in range(N):
        # Default uniform distribution on position sampling

        pos1 = np.random.choice(np.arange(math.floor(seq_len / 2)), size=1,
                                      replace=False)
        pos2 = np.random.choice(np.arange(math.ceil(seq_len / 2), seq_len), size=1,
                                      replace=False)

        X_mask[i, pos1] = 1
        X_mask[i, pos2] = 1

        Y[i] = X_num[i, pos1] + X_num[i, pos2]
    X = torch.cat((X_num.unsqueeze(-1), X_mask.unsqueeze(-1)), dim=-1)
    return X, Y.unsqueeze(-1)


def create_copying_data(batch_size, L, K, num_classes=8):
    """
    L: T
    K: c_length
    """
    seq = np.random.randint(1, high=num_classes + 1, size=(batch_size, K))
    zeros1 = np.zeros((batch_size, L))
    zeros2 = np.zeros((batch_size, K - 1))
    zeros3 = np.zeros((batch_size, K + L))
    marker = (num_classes + 1) * np.ones((batch_size, 1))

    x = torch.FloatTensor(np.concatenate((seq, zeros1, marker, zeros2), axis=1))
    y = torch.LongTensor(np.concatenate((zeros3, seq), axis=1))

    return x.unsqueeze(-1), y


def get_batch(T, batch_size):
    add_values = torch.rand(T, batch_size, requires_grad=False)

    # Build the second sequence with one 1 in each half and 0s otherwise
    add_indices = torch.zeros_like(add_values)
    half = int(T / 2)
    for i in range(batch_size):
        first_half = np.random.randint(half)
        second_half = np.random.randint(half, T)
        add_indices[first_half, i] = 1
        add_indices[second_half, i] = 1

    inputs = torch.stack((add_values, add_indices), dim=-1)
    targets = torch.mul(add_values, add_indices).sum(dim=0)
    return inputs, targets.unsqueeze(1)


# if __name__ == '__main__':
#     a, b = get_batch(200, 50)
#     print(a.shape, b.shape)