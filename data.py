import h5py
import numpy as np
import torch
import random
import os
import shutil
import glob
import cv2
from torchvision import datasets
import torchvision.transforms as T
from PIL import Image
import scipy
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch.nn.functional as F
from utils_crowd.densitymap_generator import generate_density_map, load_density, cal_new_size
from utils_crowd.kaggle_config import *
import requests
import zipfile


class Data:
    def __init__(self, X_train, Y_train, X_test, Y_test, handler, args_task):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.handler = handler
        self.args_task = args_task

        self.n_pool = len(X_train)
        self.n_test = len(X_test)

        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)

    def initialize_labels(self, num):
        # generate initial labeled pool
        tmp_idxs = np.arange(self.n_pool)
        np.random.shuffle(tmp_idxs)
        self.labeled_idxs[tmp_idxs[:num]] = True

    def get_unlabeled_data_by_idx(self, idx):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return self.X_train[unlabeled_idxs][idx]

    def get_data_by_idx(self, idx):
        return self.X_train[idx], self.Y_train[idx]

    def get_new_data(self, X, Y):
        return self.handler(X, Y, self.args_task['transform_train'])

    def get_labeled_data(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        return labeled_idxs, self.handler(self.X_train[labeled_idxs], self.Y_train[labeled_idxs],
                                          self.args_task['transform_train'])

    def get_unlabeled_data(self):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return unlabeled_idxs, self.handler(self.X_train[unlabeled_idxs], self.Y_train[unlabeled_idxs],
                                            self.args_task['transform_train'])

    def get_unlabeled_data_vaal(self):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return unlabeled_idxs, self.handler(self.X_train[unlabeled_idxs], self.Y_train[unlabeled_idxs],
                                            self.args_task['transform_train'])

    def get_unlabeled_data_subset(self, subset_size):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        random.shuffle(unlabeled_idxs)
        return unlabeled_idxs[:subset_size], self.handler(self.X_train[unlabeled_idxs[:subset_size]],
                                                          self.Y_train[unlabeled_idxs[:subset_size]],
                                                          self.args_task['transform_train'])

    def get_train_data(self):
        return self.labeled_idxs.copy(), self.handler(self.X_train, self.Y_train, self.args_task['transform_train'])

    def get_test_data(self):
        return self.handler(self.X_test, self.Y_test, self.args_task['transform'])

    def get_partial_labeled_data(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        return self.X_train[labeled_idxs], self.Y_train[labeled_idxs]

    def get_partial_unlabeled_data(self):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return self.X_train[unlabeled_idxs], self.Y_train[unlabeled_idxs]

    def cal_test_acc(self, preds):
        res = {'acc': 1.0 * (self.Y_test == preds).sum().item() / self.n_test}
        return res


class Data_Crowd_Counting(Data):
    def __init__(self, X_train, Y_train, Y_train_count, X_test, Y_test, Y_test_count, handler, args_task):
        super(Data_Crowd_Counting, self).__init__(X_train, Y_train, X_test, Y_test, handler, args_task)
        self.Y_train_count = Y_train_count
        self.Y_test_count = Y_test_count

    def cal_test_acc(self, preds):
        # MAE
        return np.mean([np.abs(self.Y_test[i].sum() - preds[i].sum()) for i in range(len(preds))])

    def get_labeled_data(self):
        # Due to the different size of the images, we need to overwrite the previous function
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        return labeled_idxs, self.handler([self.X_train[idx] for idx in labeled_idxs],
                                          [self.Y_train[idx] for idx in labeled_idxs],
                                          [self.Y_train_count[idx] for idx in labeled_idxs],
                                          self.args_task['transform_train'])

    def get_unlabeled_data(self):
        # Due to the different size of the images, we need to overwrite the previous function
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return unlabeled_idxs, self.handler([self.X_train[idx] for idx in unlabeled_idxs],
                                            [self.Y_train[idx] for idx in unlabeled_idxs],
                                            [self.Y_train_count[idx] for idx in unlabeled_idxs],
                                            self.args_task['transform_train'],
                                            validation=True)

    def get_unlabeled_data_vaal(self):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return unlabeled_idxs, self.handler([self.X_train[idx] for idx in unlabeled_idxs],
                                            [self.Y_train[idx] for idx in unlabeled_idxs],
                                            [self.Y_train_count[idx] for idx in unlabeled_idxs],
                                            self.args_task['transform_train'],
                                            resize=True)

    def get_unlabeled_data_no_resize(self):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return unlabeled_idxs, self.handler([self.X_train[idx] for idx in unlabeled_idxs],
                                            [self.Y_train[idx] for idx in unlabeled_idxs],
                                            [self.Y_train_count[idx] for idx in unlabeled_idxs],
                                            self.args_task['transform_train'])

    def get_train_data(self):
        return self.handler(self.X_train, self.Y_train, self.Y_train_count, self.args_task['transform_train'])

    def get_unlabeled_data_subset(self, subset_size):
        # Due to the different size of the images, we need to overwrite the previous function
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        random.shuffle(unlabeled_idxs)
        return unlabeled_idxs[:subset_size], self.handler([self.X_train[idx] for idx in unlabeled_idxs[:subset_size]],
                                                          [self.Y_train[idx] for idx in unlabeled_idxs[:subset_size]],
                                                          [self.Y_train_count[idx] for idx in
                                                           unlabeled_idxs[:subset_size]],
                                                          self.args_task['transform_train'])

    def get_partial_labeled_data(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        return [self.X_train[idx] for idx in labeled_idxs], [self.Y_train[idx] for idx in labeled_idxs]

    def get_partial_unlabeled_data(self):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return [self.X_train[idx] for idx in unlabeled_idxs], [self.Y_train[idx] for idx in unlabeled_idxs]

    def augment_data_pssw(self, unlabeled_data=False):
        # In the PSSW paper, they augment the data to a fixed size, so we do the same here.
        resulting_data = []
        target_augmentation = 2
        if not unlabeled_data:
            idxs = np.arange(self.n_pool)[self.labeled_idxs]
        else:
            idxs = np.arange(self.n_pool)[~self.labeled_idxs]

        num_augment = target_augmentation // len(idxs)
        augment_left = target_augmentation % len(idxs)
        resulting_idx = np.tile(idxs, num_augment)
        if augment_left > 0:
            resulting_idx = np.concatenate((resulting_idx, idxs[:augment_left]))
        return self.handler([self.X_train[idx] for idx in resulting_idx],
                            [self.Y_train[idx] for idx in resulting_idx],
                            self.args_task['transform_train'],
                            need_crop=True,
                            aug=True)

    def get_test_data(self):
        return self.handler(self.X_test, self.Y_test, self.Y_test_count, self.args_task['transform'], validation=True)

    # def get_test_data_jhu(self):
    #     return self.handler(self.X_test, self.Y_test, self.args_task['transform'], validation=True)


def get_MNIST(handler, args_task):
    raw_train = datasets.MNIST('./data/MNIST', train=True, download=True)
    raw_test = datasets.MNIST('./data/MNIST', train=False, download=True)
    return Data(raw_train.data, raw_train.targets, raw_test.data, raw_test.targets, handler, args_task)


def get_FashionMNIST(handler, args_task):
    raw_train = datasets.FashionMNIST('./data/FashionMNIST', train=True, download=True)
    raw_test = datasets.FashionMNIST('./data/FashionMNIST', train=False, download=True)
    return Data(raw_train.data, raw_train.targets, raw_test.data, raw_test.targets, handler, args_task)


def get_EMNIST(handler, args_task):
    raw_train = datasets.EMNIST('./data/EMNIST', split='byclass', train=True, download=True)
    raw_test = datasets.EMNIST('./data/EMNIST', split='byclass', train=False, download=True)
    return Data(raw_train.data, raw_train.targets, raw_test.data, raw_test.targets, handler, args_task)


def get_SVHN(handler, args_task):
    data_train = datasets.SVHN('./data/SVHN', split='train', download=True)
    data_test = datasets.SVHN('./data/SVHN', split='test', download=True)
    return Data(data_train.data, torch.from_numpy(data_train.labels), data_test.data,
                torch.from_numpy(data_test.labels), handler, args_task)


def get_CIFAR10(handler, args_task):
    data_train = datasets.CIFAR10('./data/CIFAR10', train=True, download=True)
    data_test = datasets.CIFAR10('./data/CIFAR10', train=False, download=True)
    return Data(data_train.data, torch.LongTensor(data_train.targets), data_test.data,
                torch.LongTensor(data_test.targets), handler, args_task)


def get_CIFAR10_imb(handler, args_task):
    data_train = datasets.CIFAR10('./data/CIFAR10', train=True, download=True)
    data_test = datasets.CIFAR10('./data/CIFAR10', train=False, download=True)
    X_tr = data_train.data
    Y_tr = torch.from_numpy(np.array(data_train.targets)).long()
    X_te = data_test.data
    Y_te = torch.from_numpy(np.array(data_test.targets)).long()
    ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    X_tr_imb = []
    Y_tr_imb = []
    random.seed(4666)
    for i in range(Y_tr.shape[0]):
        tmp = random.random()
        if tmp < ratio[Y_tr[i]]:
            X_tr_imb.append(X_tr[i])
            Y_tr_imb.append(Y_tr[i])
    X_tr_imb = np.array(X_tr_imb).astype(X_tr.dtype)
    Y_tr_imb = torch.LongTensor(np.array(Y_tr_imb)).type_as(Y_tr)
    return Data(X_tr_imb, Y_tr_imb, X_te, Y_te, handler, args_task)


def get_CIFAR100(handler, args_task):
    data_train = datasets.CIFAR100('./data/CIFAR100', train=True, download=True)
    data_test = datasets.CIFAR100('./data/CIFAR100', train=False, download=True)
    return Data(data_train.data, torch.LongTensor(data_train.targets), data_test.data,
                torch.LongTensor(data_test.targets), handler, args_task)


def get_TinyImageNet(handler, args_task):
    import cv2
    # download data from http://cs231n.stanford.edu/tiny-imagenet-200.zip and unzip it into ./data/TinyImageNet
    # deal with training set
    Y_train_t = []
    train_img_names = []
    train_imgs = []

    with open('./data/TinyImageNet/tiny-imagenet-200/wnids.txt') as wnid:
        for line in wnid:
            Y_train_t.append(line.strip('\n'))
    for Y in Y_train_t:
        Y_path = './data/TinyImageNet/tiny-imagenet-200/train/' + Y + '/' + Y + '_boxes.txt'
        train_img_name = []
        with open(Y_path) as Y_p:
            for line in Y_p:
                train_img_name.append(line.strip('\n').split('\t')[0])
        train_img_names.append(train_img_name)
    train_labels = np.arange(200)
    idx = 0
    for Y in Y_train_t:
        train_img = []
        for img_name in train_img_names[idx]:
            img_path = os.path.join('./data/TinyImageNet/tiny-imagenet-200/train/', Y, 'images', img_name)
            train_img.append(cv2.imread(img_path))
        train_imgs.append(train_img)
        idx = idx + 1
    train_imgs = np.array(train_imgs)
    train_imgs = train_imgs.reshape(-1, 64, 64, 3)
    X_tr = []
    Y_tr = []
    for i in range(train_imgs.shape[0]):
        Y_tr.append(i // 500)
        X_tr.append(train_imgs[i])
    # X_tr = torch.from_numpy(np.array(X_tr))
    X_tr = np.array(X_tr)
    Y_tr = torch.from_numpy(np.array(Y_tr)).long()

    # deal with testing (val) set
    Y_test_t = []
    Y_test = []
    test_img_names = []
    test_imgs = []
    with open('./data/TinyImageNet/tiny-imagenet-200/val/val_annotations.txt') as val:
        for line in val:
            test_img_names.append(line.strip('\n').split('\t')[0])
            Y_test_t.append(line.strip('\n').split('\t')[1])
    for i in range(len(Y_test_t)):
        for i_t in range(len(Y_train_t)):
            if Y_test_t[i] == Y_train_t[i_t]:
                Y_test.append(i_t)
    test_labels = np.array(Y_test)
    test_imgs = []
    for img_name in test_img_names:
        img_path = os.path.join('./data/TinyImageNet/tiny-imagenet-200/val/images', img_name)
        test_imgs.append(cv2.imread(img_path))
    test_imgs = np.array(test_imgs)
    X_te = []
    Y_te = []

    for i in range(test_imgs.shape[0]):
        X_te.append(test_imgs[i])
        Y_te.append(Y_test[i])
    # X_te = torch.from_numpy(np.array(X_te))
    X_te = np.array(X_te)
    Y_te = torch.from_numpy(np.array(Y_te)).long()
    return Data(X_tr, Y_tr, X_te, Y_te, handler, args_task)


def get_openml(handler, args_task, selection=6):
    import openml
    from sklearn.preprocessing import LabelEncoder
    openml.config.apikey = '3411e20aff621cc890bf403f104ac4bc'
    openml.config.set_cache_directory('./data/openml/')
    ds = openml.datasets.get_dataset(selection)
    data = ds.get_data(target=ds.default_target_attribute)
    X = np.asarray(data[0])
    y = np.asarray(data[1])
    y = LabelEncoder().fit(y).transform(y)

    num_classes = int(max(y) + 1)
    nSamps, _ = np.shape(X)
    testSplit = .1
    inds = np.random.permutation(nSamps)
    X = X[inds]
    y = y[inds]

    split = int((1. - testSplit) * nSamps)
    while True:
        inds = np.random.permutation(split)
        if len(inds) > 50000: inds = inds[:50000]
        X_tr = X[:split]
        X_tr = X_tr[inds]
        X_tr = torch.Tensor(X_tr)

        y_tr = y[:split]
        y_tr = y_tr[inds]
        Y_tr = torch.Tensor(y_tr).long()

        X_te = torch.Tensor(X[split:])
        Y_te = torch.Tensor(y[split:]).long()

        if len(np.unique(Y_tr)) == num_classes: break
    return Data(X_tr, Y_tr, X_te, Y_te, handler, args_task)


def get_BreakHis(handler, args_task):
    # download data from https://www.kaggle.com/datasets/ambarish/breakhis and unzip it in data/BreakHis/
    data_dir = './data/BreakHis/BreaKHis_v1/BreaKHis_v1/histology_slides/breast'
    data = datasets.ImageFolder(root=data_dir, transform=None).imgs
    train_ratio = 0.7
    test_ratio = 0.3
    data_idx = list(range(len(data)))
    random.shuffle(data_idx)
    train_idx = data_idx[:int(len(data) * train_ratio)]
    test_idx = data_idx[int(len(data) * train_ratio):]
    X_tr = [np.array(Image.open(data[i][0])) for i in train_idx]
    Y_tr = [data[i][1] for i in train_idx]
    X_te = [np.array(Image.open(data[i][0])) for i in test_idx]
    Y_te = [data[i][1] for i in test_idx]
    X_tr = np.array(X_tr, dtype=object)
    X_te = np.array(X_te, dtype=object)
    Y_tr = torch.from_numpy(np.array(Y_tr))
    Y_te = torch.from_numpy(np.array(Y_te))
    return Data(X_tr, Y_tr, X_te, Y_te, handler, args_task)


def get_PneumoniaMNIST(handler, args_task):
    # download data from https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia and unzip it in data/PhwumniaMNIST/
    import cv2

    data_train_dir = './data/PneumoniaMNIST/chest_xray/train/'
    data_test_dir = './data/PneumoniaMNIST/chest_xray/test/'
    assert os.path.exists(data_train_dir)
    assert os.path.exists(data_test_dir)

    # train data
    train_imgs_path_0 = [data_train_dir + 'NORMAL/' + f for f in os.listdir(data_train_dir + '/NORMAL/')]
    train_imgs_path_1 = [data_train_dir + 'PNEUMONIA/' + f for f in os.listdir(data_train_dir + '/PNEUMONIA/')]
    train_imgs_0 = []
    train_imgs_1 = []
    for p in train_imgs_path_0:
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, (224, 224))
        train_imgs_0.append(im)
    for p in train_imgs_path_1:
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, (224, 224))
        train_imgs_1.append(im)
    train_labels_0 = np.zeros(len(train_imgs_0))
    train_labels_1 = np.ones(len(train_imgs_1))
    X_tr = []
    Y_tr = []
    train_imgs = train_imgs_0 + train_imgs_1
    train_labels = np.concatenate((train_labels_0, train_labels_1))
    idx_train = list(range(len(train_imgs)))
    random.seed(4666)
    random.shuffle(idx_train)
    X_tr = [train_imgs[i] for i in idx_train]
    Y_tr = [train_labels[i] for i in idx_train]
    X_tr = np.array(X_tr)
    Y_tr = torch.from_numpy(np.array(Y_tr)).long()

    # test data
    test_imgs_path_0 = [data_test_dir + 'NORMAL/' + f for f in os.listdir(data_test_dir + '/NORMAL/')]
    test_imgs_path_1 = [data_test_dir + 'PNEUMONIA/' + f for f in os.listdir(data_test_dir + '/PNEUMONIA/')]
    test_imgs_0 = []
    test_imgs_1 = []
    for p in test_imgs_path_0:
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, (224, 224))
        test_imgs_0.append(im)
    for p in test_imgs_path_1:
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, (224, 224))
        test_imgs_1.append(im)
    test_labels_0 = np.zeros(len(test_imgs_0))
    test_labels_1 = np.ones(len(test_imgs_1))
    X_te = []
    Y_te = []
    test_imgs = test_imgs_0 + test_imgs_1
    test_labels = np.concatenate((test_labels_0, test_labels_1))
    idx_test = list(range(len(test_imgs)))
    random.seed(4666)
    random.shuffle(idx_test)
    X_te = [test_imgs[i] for i in idx_test]
    Y_te = [test_labels[i] for i in idx_test]
    X_te = np.array(X_tr)
    Y_te = torch.from_numpy(np.array(Y_tr)).long()

    return Data(X_tr, Y_tr, X_te, Y_te, handler, args_task)


def get_waterbirds(handler, args_task):
    import wilds
    from torchvision import transforms
    dataset = wilds.get_dataset(dataset='waterbirds', root_dir='./data/waterbirds', download='True')
    trans = transforms.Compose([transforms.Resize([255, 255])])
    train = dataset.get_subset(split='train', transform=trans)
    test = dataset.get_subset(split='test', transform=trans)

    len_train = train.metadata_array.shape[0]
    len_test = test.metadata_array.shape[0]
    X_tr = []
    Y_tr = []
    X_te = []
    Y_te = []

    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    f = open('waterbirds.txt', 'w')

    for i in range(len_train):
        x, y, meta = train.__getitem__(i)
        img = np.array(x)
        X_tr.append(img)
        Y_tr.append(y)

    for i in range(len_test):
        x, y, meta = test.__getitem__(i)
        img = np.array(x)

        X_te.append(img)
        Y_te.append(y)
        if meta[0] == 0 and meta[1] == 0:
            f.writelines('1')  # landbird_background:land
            f.writelines('\n')
            count1 = count1 + 1
        elif meta[0] == 1 and meta[1] == 0:
            f.writelines('2')  # landbird_background:water
            count2 = count2 + 1
            f.writelines('\n')
        elif meta[0] == 0 and meta[1] == 1:
            f.writelines('3')  # waterbird_background:land
            f.writelines('\n')
            count3 = count3 + 1
        elif meta[0] == 1 and meta[1] == 1:
            f.writelines('4')  # waterbird_background:water
            f.writelines('\n')
            count4 = count4 + 1
        else:
            raise NotImplementedError
    f.close()

    Y_tr = torch.tensor(Y_tr)
    Y_te = torch.tensor(Y_te)
    X_tr = np.array(X_tr)
    X_te = np.array(X_te)
    return Data(X_tr, Y_tr, X_te, Y_te, handler, args_task)


def get_shanghaitech(handler, args_task, part='A'):
    import scipy.io as io
    from scipy.io import loadmat

    if not os.path.exists('data/ShanghaiTech'):
        print('Downloading ShanghaiTech dataset...')
        os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
        os.environ["KAGGLE_KEY"] = KAGGLE_KEY
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()

        # Download the dataset
        api.dataset_download_files("tthien/shanghaitech-with-people-density-map", path="data", unzip=True)
        # Trying to delete the redundant files
        try:
            shutil.rmtree(os.path.join(os.path.abspath(''), 'data/shanghaitech_with_people_density_map/'))
        except:
            pass
    image_train = glob.glob(f'data/ShanghaiTech/part_{part}/train_data/images/*.jpg')
    image_test = glob.glob(f'data/ShanghaiTech/part_{part}/test_data/images/*.jpg')

    gt_train = glob.glob(f'data/ShanghaiTech/part_{part}/train_data/ground-truth/*.mat')
    gt_test = glob.glob(f'data/ShanghaiTech/part_{part}/test_data/ground-truth/*.mat')

    density_train = glob.glob(f'data/ShanghaiTech/part_{part}/train_data/ground-truth-h5/*.h5')
    density_test = glob.glob(f'data/ShanghaiTech/part_{part}/test_data/ground-truth-h5/*.h5')

    # Sort the images by name index.
    image_train.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    image_test.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    gt_train.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    gt_test.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    density_train.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    density_test.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # Load the images and ground truth
    train_img = []
    train_gt = []
    train_density = []
    for i in range(len(image_train)):
        im = cv2.imread(image_train[i])
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        train_img.append(im)
        gt_density = load_density(density_train[i])
        # gt_density = cv2.resize(gt_density, (int(gt_density.shape[1] / 8), int(gt_density.shape[0] / 8)),
        #                         interpolation=cv2.INTER_CUBIC) * 64
        train_density.append(gt_density)
        gt = io.loadmat(gt_train[i])
        train_gt.append(gt['image_info'][0, 0][0, 0][0])

    test_img = []
    test_gt = []
    test_density = []
    for i in range(len(image_test)):
        im = cv2.imread(image_test[i])
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        test_img.append(im)
        gt_density = load_density(density_test[i])
        test_density.append(gt_density)
        gt = io.loadmat(gt_test[i])
        test_gt.append(gt['image_info'][0, 0][0, 0][0])

    return Data_Crowd_Counting(train_img, train_density, train_gt, test_img, test_density, test_gt, handler, args_task)


def get_JHU_Crowd(handler, args_task):
    """
    Downloading the JHU Crowd dataset from www.crowd-counting.com through the Google Drive link provided in the
    website.

    @inproceedings{sindagi2019pushing,
    title={Pushing the frontiers of unconstrained crowd counting: New dataset and benchmark method},
    author={Sindagi, Vishwanath A and Yasarla, Rajeev and Patel, Vishal M},
    booktitle={Proceedings of the IEEE International Conference on Computer Vision},
    pages={1221--1231},
    year={2019}
    }

    @article{sindagi2020jhu-crowd++,
    title={JHU-CROWD++: Large-Scale Crowd Counting Dataset and A Benchmark Method},
    author={Sindagi, Vishwanath A and Yasarla, Rajeev and Patel, Vishal M},
    journal={Technical Report},
    year={2020}
    }
    """
    if not os.path.exists('data/JHU_Crowd'):
        print('Downloading JHU Crowd dataset...')
        import gdown
        os.makedirs('data/JHU_Crowd', exist_ok=True)
        url = 'https://drive.google.com/uc?id=1pA7ZeXU3hh-1txS9lFQiCek1ts3MdBaj'
        output = 'data/JHU_Crowd/JHU_Crowd.zip'
        gdown.download(url, output, quiet=False)

        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall('data/JHU_Crowd')
        os.remove(output)

    # If the density maps are not generated, generate them.
    if not os.path.exists('data/JHU_Crowd/jhu_crowd_new'):
        def read_txt(txt_path):
            data = []

            with open(txt_path, 'r') as f:
                for line in f:
                    items = line.split()
                    first_two_columns = [float(items[0]), float(items[1])]
                    data.append(first_two_columns)

            return np.array(data)

        print('The JHU Crowd dataset is not preprocessed. Preprocessing...')
        # Save dir path
        path_save = 'data/JHU_Crowd/jhu_crowd_new'

        images_train_raw = glob.glob(f'data/JHU_Crowd/jhu_crowd_v2.0/train/images/*.jpg')
        images_test_raw = glob.glob(f'data/JHU_Crowd/jhu_crowd_v2.0/test/images/*.jpg')
        images_val_raw = glob.glob(f'data/JHU_Crowd/jhu_crowd_v2.0/val/images/*.jpg')

        img_dic = {'train': images_train_raw, 'test': images_test_raw, 'val': images_val_raw}

        images_train_raw.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
        images_test_raw.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
        images_val_raw.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))

        min_size = 512
        max_size = 2048

        for phase in ['train', 'test', 'val']:
            print("Processing {} images".format(phase))
            sub_save_dir = os.path.join(path_save, phase)
            if not os.path.exists(sub_save_dir):
                os.makedirs(sub_save_dir)
                os.makedirs(os.path.join(sub_save_dir, 'images'))
                os.makedirs(os.path.join(sub_save_dir, 'density'))

            for img_path in img_dic[phase]:
                im = Image.open(img_path)
                im_w, im_h = im.size
                anno_path = img_path.replace('images', 'gt').replace('jpg', 'txt')
                points = read_txt(anno_path)
                im_h, im_w, rr = cal_new_size(im_h, im_w, min_size, max_size)
                im = np.array(im)
                if rr != 1.0:
                    im = cv2.resize(np.array(im), (im_w, im_h), cv2.INTER_CUBIC)
                    points = points * rr
                im = Image.fromarray(im)
                # Save image
                im.save(img_path.replace('jhu_crowd_v2.0', 'jhu_crowd_new'), quality=95)

                # Generate density map
                positions = generate_density_map(shape=(im_h, im_w), points=np.array(points), f_sz=15, sigma=4)
                with h5py.File(img_path.replace('jhu_crowd_v2.0', 'jhu_crowd_new').replace('images', 'density')
                                       .replace('.jpg', '.h5'), 'w') as hf:
                    hf.create_dataset("density", data=positions)
                np.save(img_path.replace('jhu_crowd_v2.0', 'jhu_crowd_new').replace('images', 'density')
                        .replace('.jpg', '.npy'), points)

        # Remove the old dataset
        shutil.rmtree('data/JHU_Crowd/jhu_crowd_v2.0')

    images_train = glob.glob('data/JHU_Crowd/jhu_crowd_new/train/images/*.jpg')
    images_test = glob.glob('data/JHU_Crowd/jhu_crowd_new/test/images/*.jpg')
    images_val = glob.glob('data/JHU_Crowd/jhu_crowd_new/val/images/*.jpg')

    images_train.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
    images_test.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
    images_val.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))

    density_train = glob.glob('data/JHU_Crowd/jhu_crowd_new/train/density/*.h5')
    density_test = glob.glob('data/JHU_Crowd/jhu_crowd_new/test/density/*.h5')
    density_val = glob.glob('data/JHU_Crowd/jhu_crowd_new/val/density/*.h5')

    print('Loading JHU Crowd dataset...')
    density_train.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
    density_test.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
    density_val.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))

    points_train = glob.glob('data/JHU_Crowd/jhu_crowd_new/train/density/*.npy')
    points_test = glob.glob('data/JHU_Crowd/jhu_crowd_new/test/density/*.npy')
    points_val = glob.glob('data/JHU_Crowd/jhu_crowd_new/val/density/*.npy')

    points_train.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
    points_test.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
    points_val.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))

    return Data_Crowd_Counting(images_train, density_train, points_train, images_test, density_test, points_test, handler, args_task)


def get_ucf_qnrf(handler, args_task):
    url_ucf_qnrf = "https://www.crcv.ucf.edu/data/ucf-qnrf/UCF-QNRF_ECCV18.zip"
    path_ucf_qnrf = "data/UCF-QNRF_ECCV18"
    path_ucf_qnrf_processed = "data/UCF-QNRF"
    if not os.path.exists(path_ucf_qnrf) and not os.path.exists(path_ucf_qnrf_processed):
        print('Downloading UCF-QNRF dataset...')
        response = requests.get(url_ucf_qnrf, stream=True, verify=False)  # notice verify=False parameter
        if response.status_code == 200:
            with open("data/UCF-QNRF_ECCV18.zip", "wb") as handle:
                for data in tqdm(response.iter_content()):
                    handle.write(data)
        else:
            print('Error while downloading UCF-QNRF dataset.')
            if os.path.exists('data/UCF-QNRF_ECCV18.zip'):
                os.remove('data/UCF-QNRF_ECCV18.zip')
            exit(1)

        # Unzip the dataset
        print('Unzipping UCF-QNRF dataset...')
        with zipfile.ZipFile('data/UCF-QNRF_ECCV18.zip', 'r') as zip_ref:
            zip_ref.extractall('data/')

        os.remove('data/UCF-QNRF_ECCV18.zip')

    if not os.path.exists(path_ucf_qnrf_processed):
        # Check the txt files
        if not os.path.exists('data_config/ucf_train.txt') or not os.path.exists('data_config/ucf_val.txt'):
            raise Exception('Missing UCF-QNRF txt files to split dataset. Please check the data_config folder.')
        print('The UCF-QNRF Crowd dataset is not preprocessed. Preprocessing...')
        # Save dir path
        path_save = 'data/UCF-QNRF'

        os.mkdir(path_save)

        images_train_raw = glob.glob(f'data/UCF-QNRF_ECCV18/Train/*.jpg')
        images_test_raw = glob.glob(f'data/UCF-QNRF_ECCV18/Test/*.jpg')

        images_train_raw.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        images_test_raw.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

        gt_train_raw = glob.glob('data/UCF-QNRF_ECCV18/Train/*.mat')
        gt_test_raw = glob.glob('data/UCF-QNRF_ECCV18/Train/*.mat')

        gt_train_raw.sort(key=lambda x: int(x.split('_')[-2]))
        gt_test_raw.sort(key=lambda x: int(x.split('_')[-2]))

        min_size = 512
        max_size = 2048

        def generate_data(img_data_path, gt_path, phase, subphase):
            for idx, img_path in enumerate(img_data_path):
                img_path_original = img_path.replace('UCF-QNRF', 'UCF-QNRF_ECCV18').replace(subphase, phase)
                anno_path_original = gt_path[idx].replace('UCF-QNRF', 'UCF-QNRF_ECCV18').replace(subphase,
                                                                                                 phase).replace(
                    '.h5', '.mat')
                im = Image.open(img_path_original)
                im_w, im_h = im.size
                points = scipy.io.loadmat(anno_path_original)['annPoints']
                im_h, im_w, rr = cal_new_size(im_h, im_w, min_size, max_size)
                im = np.array(im)
                if rr != 1.0:
                    im = cv2.resize(np.array(im), (im_w, im_h), cv2.INTER_CUBIC)
                    points = points * rr
                im = Image.fromarray(im)
                # Save image
                im.save(img_path, quality=95)

                # Generate density map
                positions = generate_density_map(shape=(im_h, im_w), points=np.array(points), f_sz=15, sigma=4)
                with h5py.File(gt_path[idx], 'w') as hf:
                    hf.create_dataset("density", data=positions)

                # Save the annPoints
                np.save(gt_path[idx].replace('.h5', '.npy'), points)

        for phase in ['Train', 'Test']:
            print("Processing {} images".format(phase))
            if phase == 'Train':
                # further split training set into training and validation set
                subphase = ['train', 'val']
                for sub in subphase:
                    if not os.path.exists(os.path.join(path_save, sub)):
                        os.mkdir(os.path.join(path_save, sub))

                    sub_path = os.path.join(path_save, sub) + '/'
                    with open(f'data_config/ucf_{sub}.txt', 'r') as f:
                        lines = f.readlines()
                        images_train_raw = [sub_path + line.strip() for line in lines]
                        gt_raw = [sub_path + line.replace('.jpg', '_ann.h5').strip() for line in lines]
                        generate_data(images_train_raw, gt_raw, phase, sub)
            else:
                if not os.path.exists(os.path.join(path_save, 'test')):
                    os.mkdir(os.path.join(path_save, 'test'))
                images_test_path = [raw.replace('UCF-QNRF_ECCV18', 'UCF-QNRF').replace('Test', 'test') for raw in
                                    images_test_raw]
                gt_test_raw = [
                    raw.replace('UCF-QNRF_ECCV18', 'UCF-QNRF').replace('Test', 'test').replace('.jpg', '_ann.h5') for
                    raw in images_test_raw]
                generate_data(images_test_path, gt_test_raw, phase, 'test')

        # Remove old
        shutil.rmtree('data/UCF-QNRF_ECCV18')

    img_train = glob.glob('data/UCF-QNRF/train/*.jpg')
    img_val = glob.glob('data/UCF-QNRF/val/*.jpg')
    img_test = glob.glob('data/UCF-QNRF/test/*.jpg')
    img_train.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    img_val.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    img_test.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    density_train = glob.glob('data/UCF-QNRF/train/*.h5')
    density_val = glob.glob('data/UCF-QNRF/val/*.h5')
    density_test = glob.glob('data/UCF-QNRF/test/*.h5')

    density_train.sort(key=lambda x: int(x.split('_')[-2]))
    density_val.sort(key=lambda x: int(x.split('_')[-2]))
    density_test.sort(key=lambda x: int(x.split('_')[-2]))

    points_train = glob.glob('data/UCF-QNRF/train/*.npy')
    points_val = glob.glob('data/UCF-QNRF/val/*.npy')
    points_test = glob.glob('data/UCF-QNRF/test/*.npy')

    points_train.sort(key=lambda x: int(x.split('_')[-2]))
    points_val.sort(key=lambda x: int(x.split('_')[-2]))
    points_test.sort(key=lambda x: int(x.split('_')[-2]))

    return Data_Crowd_Counting(img_train, density_train, points_train, img_test, density_test, points_test, handler,
                               args_task)


def get_ioc_fish(handler, args_task):
    """
    Downloading the IOCfish5K dataset from https://github.com/GuoleiSun/Indiscernible-Object-Counting through the Google
    Drive link provided in the repository.

    @inproceedings{sun2023ioc,
    title={Indiscernible Object Counting in Underwater Scenes},
    author={Sun, Guolei and An, Zhaochong and Liu, Yun and Liu, Ce and Sakaridis, Christos and Fan, Deng-Ping and Van Gool, Luc},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision and Patern Recognition (CVPR)},
    year={2023}
    }
    """
    if not os.path.exists('data/ioc_fish'):
        print('Downloading IOC Crowd dataset...')
        import gdown
        os.makedirs('data/ioc_fish_raw', exist_ok=True)
        url = 'https://drive.google.com/uc?id=1ETY_AdJB9azzja6L9URN58KtL4OH98SL'
        output = 'data/ioc_fish_raw/ioc_fish.zip'
        gdown.download(url, output, quiet=False)

        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall('data/ioc_fish_raw')
        os.remove(output)

    # # If the density maps are not generated, generate them.
    if not os.path.exists('data/ioc_fish'):
        import xml.etree.ElementTree as ET
        def extract_points_from_file(filename):
            tree = ET.parse(filename)
            root = tree.getroot()

            points = root.findall(".//object/point")

            # Obtain x, y coordinates of points
            coordinates = [(int(point.find('x').text), int(point.find('y').text)) for point in points]

            return coordinates

        print('The IOCfish5K dataset is not preprocessed. Preprocessing...')
        # Save dir path
        path_save = 'data/ioc_fish'
        os.makedirs(path_save, exist_ok=True)

        images_raw = glob.glob('data/ioc_fish_raw/released-dataset/images/*.jpg')
        gt_raw = glob.glob('data/ioc_fish_raw/released-dataset/annotations/*.xml')

        images_raw.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
        gt_raw.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))

        min_size = 512
        max_size = 2048

        def generate_data(img_data_path, gt_path, phase):
            for idx, img_path in enumerate(img_data_path):
                img_path_original = img_path.replace(f'ioc_fish/{phase}', 'ioc_fish_raw/released-dataset')
                anno_path_original = gt_path[idx].replace(f'ioc_fish/{phase}', 'ioc_fish_raw/released-dataset').replace('h5', 'xml')
                im = Image.open(img_path_original)
                im_w, im_h = im.size
                points = extract_points_from_file(anno_path_original)
                im_h, im_w, rr = cal_new_size(im_h, im_w, min_size, max_size)
                im = np.array(im)
                if rr != 1.0:
                    im = cv2.resize(np.array(im), (im_w, im_h), cv2.INTER_CUBIC)
                    points = points * rr
                im = Image.fromarray(im)
                # Save image
                im.save(img_path, quality=95)

                # Generate density map
                positions = generate_density_map(shape=(im_h, im_w), points=np.array(points), f_sz=15, sigma=4)
                with h5py.File(gt_path[idx].replace('.xml', '.h5'), 'w') as hf:
                    hf.create_dataset("density", data=positions)

                # Save the annPoints
                np.save(gt_path[idx].replace('.xml', '.npy'), points)

        for phase in ['train', 'val', 'test']:
            print("Processing {} images".format(phase))

            if not os.path.exists(os.path.join(path_save, phase)):
                os.mkdir(os.path.join(path_save, phase))
                os.mkdir(os.path.join(path_save, phase) + '/images')
                os.mkdir(os.path.join(path_save, phase) + '/annotations')

                sub_path = os.path.join(path_save, phase) + '/'
                with open(f'data_config/ioc_fish_{phase}_id.txt', 'r') as f:
                    lines = f.readlines()
                    images_train_raw = [sub_path + 'images/' + line.strip() for line in lines]
                    gt_raw = [sub_path + 'annotations/' + line.replace('.jpg', '.xml').strip() for line in lines]
                    generate_data(images_train_raw, gt_raw, phase)

        # Remove the old dataset
        shutil.rmtree('data/ioc_fish_raw')
    #
    images_train = glob.glob('data/ioc_fish/train/images/*.jpg')
    images_test = glob.glob('data/ioc_fish/test/images/*.jpg')
    images_val = glob.glob('data/ioc_fish/val/images/*.jpg')

    images_train.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
    images_test.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
    images_val.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))

    density_train = glob.glob('data/ioc_fish/train/annotations/*.h5')
    density_test = glob.glob('data/ioc_fish/test/annotations/*.h5')
    density_val = glob.glob('data/ioc_fish/val/annotations/*.h5')

    print('Loading IOCfish5K Crowd dataset...')
    density_train.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
    density_test.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
    density_val.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))

    points_train = glob.glob('data/ioc_fish/train/annotations/*.npy')
    points_test = glob.glob('data/ioc_fish/train/annotations/*.npy')
    points_val = glob.glob('data/ioc_fish/train/annotations/*.npy')

    points_train.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
    points_test.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
    points_val.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))

    return Data_Crowd_Counting(images_train, density_train, points_train, images_test, density_test, points_test, handler, args_task)


if __name__ == '__main__':
    # get_ucf_qnrf(None, None)
    get_shanghaitech(None, None)
    # get_ioc_fish(None, None)