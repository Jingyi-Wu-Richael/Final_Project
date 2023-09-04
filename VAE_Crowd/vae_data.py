import glob
import os
import random

import cv2
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import transforms
import torchvision.transforms.functional as F
from torchvision.utils import save_image

transform_shanghai = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((512, 512)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])

transform_cifar10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5, ],
                         std=[0.5, 0.5, 0.5]),
])


def get_shanghai_dataset():
    train_img = get_shanghaitech('A', 'train')
    return ShanghaitechHandler(train_img, transform=transform_shanghai)


def get_cifar_10_dataset():
    train_img, test_img = get_CIFAR10()
    return train_img, test_img


def get_shanghai_dataset_test():
    train_img = get_shanghaitech('A', 'test')
    return ShanghaitechHandler(train_img, transform=transform_shanghai)


class CIFAR10_Handler(Dataset):
    def __init__(self, X, Y, transform):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        x = Image.fromarray(x)
        x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)


class ShanghaitechHandler(Dataset):
    def __init__(self, X, transform=None):
        # X and Y are python list not np array
        self.X = X
        self.transform = transform

    def __getitem__(self, idx):
        x = self.X[idx]

        x = self.transform(x)

        # Horizontal flip
        if random.random() < 0.5:
            x = F.hflip(x)

        return x, idx

    def __len__(self):
        return len(self.X)


def get_shanghaitech(part='A', phase='train'):
    image_train = glob.glob(f'../data/ShanghaiTech/part_{part}/{phase}_data/images/*.jpg')

    # Sort the images by name index.
    image_train.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # Load the images and ground truth
    train_img = []
    for i in range(len(image_train)):
        im = cv2.imread(image_train[i])
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        train_img.append(im)

    return train_img

# def get_jhu_crowd():
#     image_train = glob.glob(f'../data/ShanghaiTech/part_{part}/{phase}_data/images/*.jpg')

def get_CIFAR10():
    data_train = datasets.CIFAR10('./data/CIFAR10', train=True, download=True, transform=transform_cifar10)
    data_test = datasets.CIFAR10('./data/CIFAR10', train=False, download=True, transform=transform_cifar10)
    return data_train, data_test
