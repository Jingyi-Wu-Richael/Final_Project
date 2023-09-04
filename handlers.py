import numpy as np
from torch.utils.data.dataset import T
import torchvision.transforms.functional as F
from torchvision import transforms
from torch.utils.data import Dataset
import torch
from PIL import Image
import cv2
from utils_crowd.densitymap_generator import load_density, generate_density_map

import random


class MNIST_Handler(Dataset):
    def __init__(self, X, Y, transform):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        x = Image.fromarray(x.numpy(), mode='L')
        x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)


class SVHN_Handler(Dataset):
    def __init__(self, X, Y, transform):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        x = Image.fromarray(np.transpose(x, (1, 2, 0)))
        x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)


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


class openml_Handler(Dataset):
    def __init__(self, X, Y, transform):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        return x, y, index

    def __len__(self):
        return len(self.X)


class MNIST_Handler_joint(Dataset):

    def __init__(self, X_1, Y_1, X_2, Y_2, transform):
        """
        :param X_1: covariate from the first distribution
        :param Y_1: label from the first distribution
        :param X_2:
        :param Y_2:
        :param transform:
        """
        self.X1 = X_1
        self.Y1 = Y_1
        self.X2 = X_2
        self.Y2 = Y_2
        self.transform = transform

    def __len__(self):

        # returning the minimum length of two data-sets

        return max(len(self.X1), len(self.X2))

    def __getitem__(self, index):
        Len1 = len(self.Y1)
        Len2 = len(self.Y2)

        # checking the index in the range or not

        if index < Len1:
            x_1 = self.X1[index]
            y_1 = self.Y1[index]

        else:

            # rescaling the index to the range of Len1
            re_index = index % Len1

            x_1 = self.X1[re_index]
            y_1 = self.Y1[re_index]

        # checking second datasets
        if index < Len2:

            x_2 = self.X2[index]
            y_2 = self.Y2[index]

        else:
            # rescaling the index to the range of Len2
            re_index = index % Len2

            x_2 = self.X2[re_index]
            y_2 = self.Y2[re_index]

        if self.transform is not None:
            x_1 = Image.fromarray(x_1.numpy(), mode='L')
            x_1 = self.transform(x_1)

            x_2 = Image.fromarray(x_2.numpy(), mode='L')
            x_2 = self.transform(x_2)

        return index, x_1, y_1, x_2, y_2


class SVHN_Handler_joint(Dataset):

    def __init__(self, X_1, Y_1, X_2, Y_2, transform=None):
        """
        :param X_1: covariate from the first distribution
        :param Y_1: label from the first distribution
        :param X_2:
        :param Y_2:
        :param transform:
        """
        self.X1 = X_1
        self.Y1 = Y_1
        self.X2 = X_2
        self.Y2 = Y_2
        self.transform = transform

    def __len__(self):

        # returning the minimum length of two data-sets

        return max(len(self.X1), len(self.X2))

    def __getitem__(self, index):
        Len1 = len(self.Y1)
        Len2 = len(self.Y2)

        # checking the index in the range or not

        if index < Len1:
            x_1 = self.X1[index]
            y_1 = self.Y1[index]

        else:

            # rescaling the index to the range of Len1
            re_index = index % Len1

            x_1 = self.X1[re_index]
            y_1 = self.Y1[re_index]

        # checking second datasets
        if index < Len2:

            x_2 = self.X2[index]
            y_2 = self.Y2[index]

        else:
            # rescaling the index to the range of Len2
            re_index = index % Len2

            x_2 = self.X2[re_index]
            y_2 = self.Y2[re_index]

        if self.transform is not None:
            x_1 = Image.fromarray(np.transpose(x_1, (1, 2, 0)))
            x_1 = self.transform(x_1)

            x_2 = Image.fromarray(np.transpose(x_2, (1, 2, 0)))
            x_2 = self.transform(x_2)

        return index, x_1, y_1, x_2, y_2


class CIFAR10_Handler_joint(Dataset):

    def __init__(self, X_1, Y_1, X_2, Y_2, transform=None):
        """
        :param X_1: covariate from the first distribution
        :param Y_1: label from the first distribution
        :param X_2:
        :param Y_2:
        :param transform:
        """
        self.X1 = X_1
        self.Y1 = Y_1
        self.X2 = X_2
        self.Y2 = Y_2
        self.transform = transform

    def __len__(self):

        # returning the minimum length of two data-sets

        return max(len(self.X1), len(self.X2))

    def __getitem__(self, index):
        Len1 = len(self.Y1)
        Len2 = len(self.Y2)

        # checking the index in the range or not

        if index < Len1:
            x_1 = self.X1[index]
            y_1 = self.Y1[index]

        else:

            # rescaling the index to the range of Len1
            re_index = index % Len1

            x_1 = self.X1[re_index]
            y_1 = self.Y1[re_index]

        # checking second datasets
        if index < Len2:

            x_2 = self.X2[index]
            y_2 = self.Y2[index]

        else:
            # rescaling the index to the range of Len2
            re_index = index % Len2

            x_2 = self.X2[re_index]
            y_2 = self.Y2[re_index]

        if self.transform is not None:
            x_1 = Image.fromarray(x_1)
            x_1 = self.transform(x_1)

            x_2 = Image.fromarray(x_2)
            x_2 = self.transform(x_2)

        return index, x_1, y_1, x_2, y_2


class ShanghaitechHandler(Dataset):
    def __init__(self, X, Y, Z, transform=None, aug=False, resize=False, validation=False):
        # X and Y are python list not np array
        self.X = X
        self.Y = Y
        self.Z = Z
        self.transform = transform
        self.aug = aug
        self.resize = resize
        self.validation = validation

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        y = torch.from_numpy(y.copy()).float()

        # z = self.Z[idx]

        def random_crop(c_x, c_y):
            crop_size = (c_x.shape[0] // 2, c_x.shape[1] // 2)
            # Follows the implementation of the paper
            i = random.randint(0, c_x.shape[0] - crop_size[0])
            j = random.randint(0, c_x.shape[1] - crop_size[1])

            return c_x[i:i + crop_size[0], j:j + crop_size[1]], \
                c_y[i:i + crop_size[0], j:j + crop_size[1]]
            #
            # return c_x.crop((i, j, i + crop_size[0], j + crop_size[1])), \
            #     c_y.crop((i, j, i + crop_size[0], j + crop_size[1]))

        def paired_crop(c_x, c_y):
            w, h, c = c_x.shape
            if w % 16 == 0 and h % 16 == 0:
                return c_x, c_y
            else:
                return c_x[:w - w % 16, :h - h % 16], c_y[:w - w % 16, :h - h % 16]

        if self.aug:
            x, y = random_crop(x, y)

        if not self.resize:
            x, y = paired_crop(x, y)
            x = self.transform(x)
        else:
            x = self.transform(x)
            x = torch.nn.functional.interpolate(x.unsqueeze(0), size=(224, 224), mode='bilinear',
                                                align_corners=True).squeeze(0)
            y = torch.nn.functional.interpolate(y.unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear',
                                                align_corners=True).squeeze(0).squeeze(0)

        # Horizontal flip
        if random.random() < 0.5:
            x = F.hflip(x)
            y = F.hflip(y)

        return x, y, idx

    def __len__(self):
        return len(self.X)


class ShanghaitechHandler_Semi(Dataset):
    def __init__(self, X1, Y1, X2, Y2, transform=None, resize=True, data_ratio=4):
        # X1 and Y1 are labelled data
        self.X1 = X1 * data_ratio
        self.Y1 = Y1 * data_ratio
        self.X2 = X2
        self.Y2 = Y2
        self.X = self.X1 + self.X2
        self.Y = self.Y1 + self.Y2
        self.transform = transform
        self.resize = resize

    def __getitem__(self, idx):
        # Check the index
        x = self.X[idx]
        y = self.Y[idx]

        labeled = idx < len(self.X1)
        y = torch.from_numpy(y.copy()).float()

        def paired_crop(c_x, c_y):
            w, h, c = c_x.shape
            if w % 16 == 0 and h % 16 == 0:
                return c_x, c_y
            else:
                return c_x[:w - w % 16, :h - h % 16], c_y[:w - w % 16, :h - h % 16]

        x, y = paired_crop(x, y)
        x = self.transform(x)

        if self.resize:
            y = torch.nn.functional.interpolate(y.unsqueeze(0).unsqueeze(0), size=(y.shape[0] // 8, y.shape[1] // 8),
                                                mode='bilinear',
                                                align_corners=True).squeeze(0).squeeze(0) * 64

        return x, y, labeled, idx

    def __len__(self):
        return len(self.X)


class ShanghaitechHandler_AE(Dataset):
    def __init__(self, X, transform=None):
        self.X = X
        self.transform = transform

    def __getitem__(self, idx):
        x = self.X[idx]
        x = self.transform(x)
        x = torch.nn.functional.interpolate(x.unsqueeze(0), size=(512, 512), mode='bilinear',
                                            align_corners=True).squeeze(0)
        return x, idx

    def __len__(self):
        return len(self.X)


class ShanghaitechHandler_joint(Dataset):
    def __init__(self, X1, Y1, X2, Y2, transform=None):
        # X and Y are python list not np array
        self.X1 = X1
        self.Y1 = Y1
        self.X2 = X2
        self.Y2 = Y2
        self.transform = transform

    def __getitem__(self, idx):
        # Check the index
        if idx < len(self.X1):
            x1 = self.X1[idx]
            y1 = self.Y1[idx]
        else:
            x1 = self.X2[idx % len(self.X1)]
            y1 = self.Y2[idx % len(self.X1)]

        if idx < len(self.X2):
            x2 = self.X2[idx]
            y2 = self.Y2[idx]
        else:
            x2 = self.X1[idx % len(self.X2)]
            y2 = self.Y1[idx % len(self.X2)]

        # Crop both images
        x1 = self.transform(x1)
        x2 = self.transform(x2)
        y1 = torch.from_numpy(y1.copy()).float()
        y2 = torch.from_numpy(y2.copy()).float()

        # Resize to 224x224
        x1 = torch.nn.functional.interpolate(x1.unsqueeze(0), size=(224, 224), mode='bilinear',
                                             align_corners=True).squeeze(0)
        x2 = torch.nn.functional.interpolate(x2.unsqueeze(0), size=(224, 224), mode='bilinear',
                                             align_corners=True).squeeze(0)
        y1 = torch.nn.functional.interpolate(y1.unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear',
                                             align_corners=True).squeeze(0).squeeze(0)
        y2 = torch.nn.functional.interpolate(y2.unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear',
                                             align_corners=True).squeeze(0).squeeze(0)

        # Horizontal flip
        if random.random() < 0.5:
            x1 = F.hflip(x1)
            y1 = F.hflip(y1)

        if random.random() < 0.5:
            x2 = F.hflip(x2)
            y2 = F.hflip(y2)

        return idx, x1, y1, x2, y2

    def __len__(self):
        return max((len(self.X1), len(self.X2)))


class JHUCrowdHandler(Dataset):
    def __init__(self, X, Y, Z, transform=None, aug=False, resize=False, validation=False):
        # X and Y are python list not np array
        self.X = X
        self.Y = Y
        self.Z = Z
        self.transform = transform
        self.aug = aug
        self.resize = resize
        self.max_size = 2048
        self.validation = validation

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]

        # Load image
        x = cv2.imread(x)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

        # Load density map
        y = load_density(y)

        y = torch.from_numpy(y).float()

        def random_crop(c_x, c_y, crop_size):
            # Follows the implementation of the paper
            i = random.randint(0, c_x.shape[0] - crop_size[0])
            j = random.randint(0, c_x.shape[1] - crop_size[1])

            return c_x[i:i + crop_size[0], j:j + crop_size[1]], \
                c_y[i:i + crop_size[0], j:j + crop_size[1]]
            #
            # return c_x.crop((i, j, i + crop_size[0], j + crop_size[1])), \
            #     c_y.crop((i, j, i + crop_size[0], j + crop_size[1]))

        def paired_crop(c_x, c_y):
            w, h, c = c_x.shape
            if w % 16 == 0 and h % 16 == 0:
                return c_x, c_y
            else:
                return c_x[:w - w % 16, :h - h % 16], c_y[:w - w % 16, :h - h % 16]

        # if not self.validation:
        #     width, height, _ = x.shape
        #     if max(x.shape[0], x.shape[1]) > self.max_size:
        #         if width > height:
        #             new_width = self.max_size
        #             new_height = int(self.max_size * height / width)
        #         else:
        #             new_height = self.max_size
        #             new_width = int(self.max_size * width / height)
        #     # crop
        #         x, y = random_crop(x, y, (new_width, new_height))
            # x, y = random_crop(x, y, (256, 256))

        x, y = paired_crop(x, y)
        x = self.transform(x)

        # Horizontal flip
        if random.random() < 0.5:
            x = F.hflip(x)
            y = F.hflip(y)

        return x, y, idx

    def __len__(self):
        return len(self.X)


class SGANetHandler(Dataset):
    """
    Dataset handler for SGANet:
    https://github.com/hellowangqian/sganet-crowd-counting/blob/master/headCounting_shanghaitech_segLoss.py#L78
    """

    def __init__(self, X, Y, Z, transform=None, validation=False, patch_size=128, num_patches_per_image=4):
        # X and Y are python list not np array
        self.X = X
        self.Y = Y
        self.Z = Z
        self.transform = transform
        self.validation = validation
        self.patch_size = patch_size
        self.num_patches = num_patches_per_image

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # check type
        needLoad = isinstance(self.X[idx], str)

        if needLoad:
            image = cv2.imread(self.X[idx])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # H,W,C
            positions = self.Y[idx]
            positions = load_density(positions)
            annPoints = np.load(self.Z[idx])
        else:
            image = self.X[idx]
            positions = self.Y[idx]
            positions = torch.from_numpy(positions.copy()).float()
            annPoints = self.Z[idx]

        fbs = generate_density_map(shape=image.shape, points=annPoints, f_sz=25, sigma=1)
        fbs = np.int32(fbs>0)
        targetSize = [self.patch_size, self.patch_size]
        height, width, channel = image.shape
        if height < targetSize[0] or width < targetSize[1]:
            image = cv2.resize(image, (np.maximum(targetSize[0] + 2, height), np.maximum(targetSize[1] + 2, width)))
            count = positions.sum()
            max_value = positions.max()
            # down density map
            positions = cv2.resize(positions,
                                   (np.maximum(targetSize[0] + 2, height), np.maximum(targetSize[1] + 2, width)))
            count2 = positions.sum()
            positions = np.minimum(positions * count / (count2 + 1e-8), max_value * 10)
            fbs = cv2.resize(fbs, (np.maximum(targetSize[0] + 2, height), np.maximum(targetSize[1] + 2, width)))
            fbs = np.int32(fbs > 0)
        if len(image.shape) == 2:
            image = np.expand_dims(image, 2)
            image = np.concatenate((image, image, image), axis=2)
        # transpose from h x w x channel to channel x h x w
        image = image.transpose(2, 0, 1)
        numPatches = self.num_patches

        patchSet, countSet, fbsSet = [], [], []
        if not self.validation:
            patchSet, countSet, fbsSet = self.getRandomPatchesFromImage(image, positions, fbs, targetSize, numPatches)
            x = np.zeros((patchSet.shape[0], 3, targetSize[0], targetSize[1]))
            if self.transform:
                for i in range(patchSet.shape[0]):
                    # transpose to original:h x w x channel
                    x[i, :, :, :] = self.transform(np.uint8(patchSet[i, :, :, :]).transpose(1, 2, 0))
            patchSet = x
        if self.validation:
            patchSet, countSet, fbsSet = self.getAllFromImage(image, positions, fbs)
            patchSet[0, :, :, :] = self.transform(np.uint8(patchSet[0, :, :, :]).transpose(1, 2, 0))
            return patchSet, countSet, idx
        return patchSet, countSet, fbsSet, idx

    def getRandomPatchesFromImage(self, image, positions, fbs, target_size, numPatches):
        # generate random cropped patches with pre-defined size, e.g., 224x224
        imageShape = image.shape
        if np.random.random() > 0.5:
            for channel in range(3):
                image[channel, :, :] = np.fliplr(image[channel, :, :])
            positions = np.fliplr(positions)
            fbs = np.fliplr(fbs)
        patchSet = np.zeros((numPatches, 3, target_size[0], target_size[1]))
        # generate density map
        countSet = np.zeros((numPatches, 1, target_size[0], target_size[1]))
        fbsSet = np.zeros((numPatches, 1, target_size[0], target_size[1]))
        for i in range(numPatches):
            topLeftX = np.random.randint(imageShape[1] - target_size[0] + 1)  # x-height
            topLeftY = np.random.randint(imageShape[2] - target_size[1] + 1)  # y-width
            thisPatch = image[:, topLeftX:topLeftX + target_size[0], topLeftY:topLeftY + target_size[1]]
            patchSet[i, :, :, :] = thisPatch
            # density map
            position = positions[topLeftX:topLeftX + target_size[0], topLeftY:topLeftY + target_size[1]]
            fb = fbs[topLeftX:topLeftX + target_size[0], topLeftY:topLeftY + target_size[1]]
            position = position.reshape((1, position.shape[0], position.shape[1]))
            fb = fb.reshape((1, fb.shape[0], fb.shape[1]))
            countSet[i, :, :, :] = position
            fbsSet[i, :, :, :] = fb
        return patchSet, countSet, fbsSet

    def getAllFromImage(self, image, positions, fbs):
        nchannel, height, width = image.shape
        patchSet = np.zeros((1, 3, height, width))
        patchSet[0, :, :, :] = image[:, :, :]
        countSet = positions.reshape((1, 1, positions.shape[0], positions.shape[1]))
        fbsSet = fbs.reshape((1, 1, fbs.shape[0], fbs.shape[1]))
        return patchSet, countSet, fbsSet


# ----------------------------------#
#          Transform code          #
# ----------------------------------#
# The code below is taken from https://github.com/CommissarMa/CSRNet-pytorch
class PairedCrop(object):
    '''
    Paired Crop for both image and its density map.
    Note that due to the maxpooling in the nerual network,
    we must promise that the size of input image is the corresponding factor.
    '''

    def __init__(self, factor=16):
        self.factor = factor

    @staticmethod
    def get_params(img, factor):
        w, h = img.size
        if w % factor == 0 and h % factor == 0:
            return 0, 0, h, w
        else:
            return 0, 0, h - (h % factor), w - (w % factor)

    def __call__(self, img_and_dmap):
        '''
        img: PIL.Image
        dmap: PIL.Image
        '''
        img, dmap = img_and_dmap

        i, j, th, tw = self.get_params(img, self.factor)

        img = F.crop(img, i, j, th, tw)
        dmap = F.crop(dmap, i, j, th, tw)
        return (img, dmap)
