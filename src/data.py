import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import FashionMNIST, MNIST, CIFAR10, SVHN
import torchvision
import torchvision.utils as vision_utils


class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func
        self.dataset = self.dl.dataset

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))
            
            
def keep_only_lbls(dataset, lbls, device=None):
    lbls = {lbl: i for i, lbl in enumerate(lbls)}
    final_X, final_Y = [], []
    for x, y in dataset:
        if y in lbls:
            final_X.append(x)
            final_Y.append(lbls[y])
    X = torch.stack(final_X)
    Y = torch.tensor(final_Y).long()
    if device is not None:
        X = X.to(device)
        Y = Y.to(device)
    return torch.utils.data.TensorDataset(X, Y)


def get_fashion_mnist_dl(args):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    
    data_train = FashionMNIST('./datasets', train=True, download=True, transform=transform)
    data_train = keep_only_lbls(data_train, lbls=[0,1,2,3,4,5,6,7,8,9], device=args.device)
    data_train, data_valid = torch.utils.data.random_split(data_train, [55000,5000])
    
    data_test = FashionMNIST('./datasets', train=False, download=True, transform=transform)
    data_test = keep_only_lbls(data_test, lbls=[0,1,2,3,4,5,6,7,8,9], device=args.device)
    
    train_dl = DataLoader(data_train, batch_size=args.batch_size_train, shuffle=True)
    valid_dl = DataLoader(data_valid, batch_size=args.batch_size_eval, shuffle=False)
    test_dl = DataLoader(data_test, batch_size=args.batch_size_eval, shuffle=False)
    
    return train_dl, valid_dl, test_dl


def get_mnist_dl(args):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    
    data_train = MNIST('./datasets', train=True, download=True, transform=transform)
    data_train = keep_only_lbls(data_train, lbls=[0,1,2,3,4,5,6,7,8,9], device=args.device)
    data_train, data_valid = torch.utils.data.random_split(data_train, [55000,5000])
    
    data_test = MNIST('./datasets', train=False, download=True, transform=transform)
    data_test = keep_only_lbls(data_test, lbls=[0,1,2,3,4,5,6,7,8,9], device=args.device)
    
    train_dl = DataLoader(data_train, batch_size=args.batch_size_train, shuffle=True)
    valid_dl = DataLoader(data_valid, batch_size=args.batch_size_eval, shuffle=False)
    test_dl = DataLoader(data_test, batch_size=args.batch_size_eval, shuffle=False)
    
    return train_dl, valid_dl, test_dl


def get_svhn_dl(args):
    transform = torchvision.transforms.ToTensor()
    
    data_train = SVHN('./datasets', split='train', download=True, transform=transform)
    data_train = keep_only_lbls(data_train, lbls=[0,1,2,3,4,5,6,7,8,9], device=None)
    data_train, data_valid = torch.utils.data.random_split(data_train, [67000,6257])
    
    data_test = SVHN('./datasets', split='test', download=True, transform=transform)
    data_test = keep_only_lbls(data_test, lbls=[0,1,2,3,4,5,6,7,8,9], device=None)
    
    train_dl = DataLoader(data_train, batch_size=args.batch_size_train, shuffle=True, num_workers=7, pin_memory=True)
    valid_dl = DataLoader(data_valid, batch_size=args.batch_size_eval, shuffle=False, num_workers=5, pin_memory=True)
    test_dl = DataLoader(data_test, batch_size=args.batch_size_eval, shuffle=False, num_workers=5, pin_memory=True)
    
    train_dl = WrappedDataLoader(train_dl, lambda x, y: (x.to(args.device), y.to(args.device))) # Data directly sent to GPU by dataloader
    valid_dl = WrappedDataLoader(valid_dl, lambda x, y: (x.to(args.device), y.to(args.device)))
    test_dl = WrappedDataLoader(test_dl, lambda x, y: (x.to(args.device), y.to(args.device)))
    
    return train_dl, valid_dl, test_dl


def get_cifar10_dl(args):
    transform = torchvision.transforms.ToTensor()
    
    data_train = CIFAR10(root='./datasets', train=True, download=True, transform=transform)
    data_train = keep_only_lbls(data_train, lbls=[0,1,2,3,4,5,6,7,8,9], device=None)
    data_train, data_valid = torch.utils.data.random_split(data_train, [45000,5000])
    
    data_test = CIFAR10(root='./datasets', train=False, download=True, transform=transform)
    data_test = keep_only_lbls(data_test, lbls=[0,1,2,3,4,5,6,7,8,9], device=None)

    train_dl = DataLoader(data_train, batch_size=args.batch_size_train, shuffle=True, num_workers=7, pin_memory=True)
    valid_dl = DataLoader(data_valid, batch_size=args.batch_size_eval, shuffle=False, num_workers=5, pin_memory=True)
    test_dl = DataLoader(data_test, batch_size=args.batch_size_eval, shuffle=False, num_workers=5, pin_memory=True)
    
    train_dl = WrappedDataLoader(train_dl, lambda x, y: (x.to(args.device), y.to(args.device))) # Data directly sent to GPU by dataloader
    valid_dl = WrappedDataLoader(valid_dl, lambda x, y: (x.to(args.device), y.to(args.device)))
    test_dl = WrappedDataLoader(test_dl, lambda x, y: (x.to(args.device), y.to(args.device)))
    
    return train_dl, valid_dl, test_dl


def get_cifar10_data_aug_dl(args):
    transform_test = torchvision.transforms.ToTensor()
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
    ])
    
    data_train = CIFAR10(root='./datasets', train=True, download=True, transform=transform_train)
    data_train, data_valid = torch.utils.data.random_split(data_train, [45000,5000])
    
    data_test = CIFAR10(root='./datasets', train=False, download=True, transform=transform_test)
    data_test = keep_only_lbls(data_test, lbls=[0,1,2,3,4,5,6,7,8,9], device=None)

    train_dl = DataLoader(data_train, batch_size=args.batch_size_train, shuffle=True, num_workers=7, pin_memory=True)
    valid_dl = DataLoader(data_valid, batch_size=args.batch_size_eval, shuffle=False, num_workers=5, pin_memory=True)
    test_dl = DataLoader(data_test, batch_size=args.batch_size_eval, shuffle=False, num_workers=5, pin_memory=True)
    
    train_dl = WrappedDataLoader(train_dl, lambda x, y: (x.to(args.device), y.to(args.device))) # Data directly sent to GPU by dataloader
    valid_dl = WrappedDataLoader(valid_dl, lambda x, y: (x.to(args.device), y.to(args.device)))
    test_dl = WrappedDataLoader(test_dl, lambda x, y: (x.to(args.device), y.to(args.device)))
    
    return train_dl, valid_dl, test_dl


def get_cifar10_ldr_dl(args):
    transform = torchvision.transforms.ToTensor()
    
    data_train = CIFAR10(root='./datasets', train=True, download=True, transform=transform)
    data_train = keep_only_lbls(data_train, lbls=[0,1,2,3,4,5,6,7,8,9], device=args.device)
    data_train, data_valid, _ = torch.utils.data.random_split(data_train, [2500,5000,42500], generator=torch.Generator().manual_seed(42))
    
    data_test = CIFAR10(root='./datasets', train=False, download=True, transform=transform)
    data_test = keep_only_lbls(data_test, lbls=[0,1,2,3,4,5,6,7,8,9], device=args.device)

    train_dl = DataLoader(data_train, batch_size=args.batch_size_train, shuffle=True)
    valid_dl = DataLoader(data_valid, batch_size=args.batch_size_eval, shuffle=False)
    test_dl = DataLoader(data_test, batch_size=args.batch_size_eval, shuffle=False)
    
    return train_dl, valid_dl, test_dl


def get_dataset(args):
    if args.dataset == 'fashion-mnist':
        return get_fashion_mnist_dl(args)
    elif args.dataset == 'mnist':
        return get_mnist_dl(args)
    elif args.dataset == 'svhn':
        return get_svhn_dl(args)
    elif args.dataset == 'cifar10':
        return get_cifar10_dl(args)
    elif args.dataset == 'cifar10-daug':
        return get_cifar10_data_aug_dl(args)
    elif args.dataset == 'cifar10-ldr':
        return get_cifar10_ldr_dl(args)
    else:
        raise KeyError(f"Unknown dataset '{args.dataset}'.")
    
    