"""
Code taken from https://github.com/TencentYoutuResearch/CrowdCounting-SASNet/blob/main/prepare_dataset.py
"""

import h5py
import numpy as np


# the function to generate the density map, with provided points
def generate_density_map(shape=(5, 5), points=None, f_sz=15, sigma=4):
    """
    generate density map given head coordinations
    """
    im_density = np.zeros(shape[0:2])
    h, w = shape[0:2]
    if len(points) == 0:
        return im_density
    # iterate over all the points
    for j in range(len(points)):
        # create the gaussian kernel
        H = matlab_style_gauss2D((f_sz, f_sz), sigma)
        # limit the bound
        x = np.minimum(w, np.maximum(1, np.abs(np.int32(np.floor(points[j, 0])))))
        y = np.minimum(h, np.maximum(1, np.abs(np.int32(np.floor(points[j, 1])))))
        if x > w or y > h:
            continue
        # get the rect around each head
        x1 = x - np.int32(np.floor(f_sz / 2))
        y1 = y - np.int32(np.floor(f_sz / 2))
        x2 = x + np.int32(np.floor(f_sz / 2))
        y2 = y + np.int32(np.floor(f_sz / 2))
        dx1 = 0
        dy1 = 0
        dx2 = 0
        dy2 = 0
        change_H = False
        if x1 < 1:
            dx1 = np.abs(x1) + 1
            x1 = 1
            change_H = True
        if y1 < 1:
            dy1 = np.abs(y1) + 1
            y1 = 1
            change_H = True
        if x2 > w:
            dx2 = x2 - w
            x2 = w
            change_H = True
        if y2 > h:
            dy2 = y2 - h
            y2 = h
            change_H = True
        x1h = 1 + dx1
        y1h = 1 + dy1
        x2h = f_sz - dx2
        y2h = f_sz - dy2
        if change_H:
            H = matlab_style_gauss2D((y2h - y1h + 1, x2h - x1h + 1), sigma)
        # attach the gaussian kernel to the rect of this head
        im_density[y1 - 1:y2, x1 - 1:x2] = im_density[y1 - 1:y2, x1 - 1:x2] + H
    return im_density


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def load_density(file_path):
    gt_file = h5py.File(file_path, 'r')
    groundtruth = np.asarray(gt_file['density'])
    groundtruth = groundtruth.astype(np.float32, copy=False)
    return groundtruth


def cal_new_size(im_h, im_w, min_size, max_size):
    if im_h < im_w:
        if im_h < min_size:
            ratio = 1.0 * min_size / im_h
            im_h = min_size
            im_w = round(im_w*ratio)
        elif im_h > max_size:
            ratio = 1.0 * max_size / im_h
            im_h = max_size
            im_w = round(im_w*ratio)
        else:
            ratio = 1.0
    else:
        if im_w < min_size:
            ratio = 1.0 * min_size / im_w
            im_w = min_size
            im_h = round(im_h*ratio)
        elif im_w > max_size:
            ratio = 1.0 * max_size / im_w
            im_w = max_size
            im_h = round(im_h*ratio)
        else:
            ratio = 1.0
    return im_h, im_w, ratio


def densitymap_to_densitymask(density_map, threshold):
    density_mask = (density_map > threshold).float()
    return density_mask


if __name__ == '__main__':
    import cv2
    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F
    img = cv2.imread('/Users/daizihan/Desktop/deepALplus/data/demo/0109.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gt = load_density('/Users/daizihan/Desktop/deepALplus/data/demo/0109.h5')
    gt = torch.from_numpy(gt.copy()).float()
    print(gt.sum())
    dm = densitymap_to_densitymask(gt, 0)
    print(dm.sum())