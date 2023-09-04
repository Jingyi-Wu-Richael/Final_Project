import cv2
import torch
import numpy as np
import random


def further_crop(imgs, densities):
    # We need further crop the images to make them having same size
    # min width
    min_width = min([img.shape[2] for img in imgs])
    min_height = min([img.shape[1] for img in imgs])

    # Crop the images
    imgs = [img[:, :min(min_height, img.shape[1]), :min(min_width, img.shape[2])] for img in imgs]
    densities = [density[:min_height, :min_width] for density in densities]

    # # Resize the densities
    #
    # densities = [cv2.resize(density, (int(density.shape[1] / 8), int(density.shape[0] / 8)),
    #                         interpolation=cv2.INTER_CUBIC) * 64 for density in densities]
    return imgs, densities


def pssw_collate_fn(batch):
    # Separate the data, labels and label_info
    img, labels, is_labelled = zip(*batch)

    # Convert data, labels and is_labelled to tensors
    img, labels = further_crop(img, labels)
    img = torch.stack(img)

    labels = torch.stack(labels)
    is_labelled = torch.tensor(is_labelled)

    return img, labels, is_labelled


def pssw_resize_collate_fn(batch):
    # Separate the data, labels and label_info
    img, labels, indices = zip(*batch)

    # Convert data, labels and is_labelled to tensors
    # Need to further crop the images
    img, labels = further_crop(img, labels)
    img = torch.stack(img)
    labels = torch.stack(labels)

    indices = list(indices)

    return img, labels, indices


def get_min_size(batch):
    min_ht = 576
    min_wd = 768

    for i_sample in batch:

        _, ht, wd = i_sample.shape
        if ht < min_ht:
            min_ht = ht
        if wd < min_wd:
            min_wd = wd
    return min_ht, min_wd


def random_crop(img, den, dst_size):
    # dst_size: ht, wd

    _, ts_hd, ts_wd = img.shape

    x1 = random.randint(0, ts_wd - dst_size[1])
    y1 = random.randint(0, ts_hd - dst_size[0])
    x2 = x1 + dst_size[1]
    y2 = y1 + dst_size[0]

    label_x1 = x1
    label_y1 = y1
    label_x2 = x2
    label_y2 = y2

    return img[:, y1:y2, x1:x2], den[label_y1:label_y2, label_x1:label_x2]


def SHHA_collate(batch):
    # @GJY
    r"""Puts each data field into a tensor with outer dimension batch size"""

    transposed = list(zip(*batch))  # imgs and dens
    imgs, dens, idxs = [transposed[0], transposed[1], transposed[2]]

    error_msg = "batch must contain tensors; found {}"
    if isinstance(imgs[0], torch.Tensor) and isinstance(dens[0], torch.Tensor):

        min_ht, min_wd = get_min_size(imgs)

        cropped_imgs = []
        cropped_dens = []
        for i_sample in range(len(batch)):
            _img, _den = random_crop(imgs[i_sample], dens[i_sample], [min_ht, min_wd])
            cropped_imgs.append(_img)
            cropped_dens.append(_den)

        cropped_imgs = torch.stack(cropped_imgs, 0)
        cropped_dens = torch.stack(cropped_dens, 0)

        return [cropped_imgs, cropped_dens, idxs]

    raise TypeError((error_msg.format(type(batch[0]))))