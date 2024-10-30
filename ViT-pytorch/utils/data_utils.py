import logging
from pathlib import Path

import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
import os
from torch.utils.data import Dataset
import cv2
from PIL import Image

logger = logging.getLogger(__name__)

class CustomDataset(Dataset):
    def __init__(self, args, type_train):
        self.type_train = type_train
        self.dataset_dir = args.dataset_path
        self.class_folders = [ f.name for f in os.scandir(self.dataset_dir) if f.is_dir() ]
        self.class_folders.sort()
        self.images = []
        for class_name in self.class_folders:
            class_images = []
            for name in os.listdir(os.path.join(self.dataset_dir, class_name, 'train' if type_train else 'test')):
                class_images.append(os.path.join(self.dataset_dir, class_name, 'train' if type_train else 'test', name))
            class_images.sort()
            self.images.append(class_images)
        # self.images = [[os.path.join(self.dataset_dir, folder, name) for name in os.listdir(os.path.join(self.dataset_dir, folder))] for folder in self.class_folders]
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]) if self.type_train else transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    def __len__(self):
        return sum([len(i) for i in self.images])
    def __getitem__(self, index):
        for i in range(len(self.images)):
            if(index < len(self.images[i])):
                image = Image.open(self.images[i][index]).convert('RGB') # cv2.imread(self.images[i][index])
                image = self.transform(image)
                return image, i
            else:
                index -= len(self.images[i])
        return None

def get_loader_custom(args):
    # dataset = CustomDataset(args.dataset_path)
    # logger.info([dataset[i] for i in range(len(dataset))])
    trainset = CustomDataset(args, True)
    testset = CustomDataset(args, False)
    
    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None
    
    return train_loader, test_loader
    
    # dataset_path = args.dataset_path
    # subfolders = [ f.name for f in os.scandir(dataset_path) if f.is_dir() ]
    # logger.info(subfolders)

def get_loader(args):
    return get_loader_custom(args)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if args.dataset == "cifar10":
        trainset = datasets.CIFAR10(root="./data",
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        testset = datasets.CIFAR10(root="./data",
                                   train=False,
                                   download=True,
                                   transform=transform_test) if args.local_rank in [-1, 0] else None

    else:
        trainset = datasets.CIFAR100(root="./data",
                                     train=True,
                                     download=True,
                                     transform=transform_train)
        testset = datasets.CIFAR100(root="./data",
                                    train=False,
                                    download=True,
                                    transform=transform_test) if args.local_rank in [-1, 0] else None
    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader
