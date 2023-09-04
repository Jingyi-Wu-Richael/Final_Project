import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from nets import Net
import random
import math
import cv2
import numpy as np


class GaussianNLLLossLogVar(nn.Module):
    def __init__(self, full=False, eps=1e-6, reduction='mean'):
        super().__init__()
        self.full = full
        self.eps = eps
        self.reduction = reduction

    def forward(self, input, target, log_var):
        log_var = log_var.clamp(min=self.eps)
        loss = 0.5 * (log_var + ((input - target) ** 2) / torch.exp(log_var))
        if self.full:
            loss += 0.5 * math.log(2 * math.pi)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


class Net_Crowd_Counting_Project_DUBNet(Net):
    def __init__(self, net, params, device):
        super(Net_Crowd_Counting_Project_DUBNet, self).__init__(net, params, device)

    def train(self, labelled_data, test_data):
        n_epoch = self.params['n_epoch']

        self.clf = self.net()

        if torch.cuda.device_count() > 1:
            self.clf = nn.DataParallel(self.clf)
        self.clf.to(self.device)

        self.clf.train()
        if self.params['optimizer'] == 'Adam':
            optimizer = optim.Adam(self.clf.parameters(), **self.params['optimizer_args'])
            lr_schedule = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=1)
        elif self.params['optimizer'] == 'SGD':
            optimizer = optim.SGD(self.clf.parameters(), **self.params['optimizer_args'])
            lr_schedule = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)
        else:
            raise NotImplementedError

        loader_labelled = DataLoader(labelled_data, shuffle=True, **self.params['loader_tr_args'])

        min_mae = sys.maxsize

        best_epoch = 0
        # criterion = GaussianNLLLossLogVar(reduction='mean')
        criterion = nn.GaussianNLLLoss(reduction='mean')
        for epoch in tqdm(range(1, n_epoch + 1), ncols=100):
            if self.params['optimizer'] == 'Adam':
                lr_schedule.step()
            # Training labelled data
            for batch_idx, (x, y, idxs) in enumerate(loader_labelled):
                x, y = x.to(self.device), y.to(self.device)
                y = torch.nn.functional.interpolate(y.unsqueeze(0), size=(y.shape[1] // 16, y.shape[2] // 16),
                                                    mode='bilinear', align_corners=True) * 16 * 16
                optimizer.zero_grad()
                density, uncertainty = self.clf(x)
                nll = criterion(density, y, uncertainty)

                # Calculate total loss
                total_loss = nll
                print('log_var:', uncertainty.mean().item(), 'nll:', nll.item())

                total_loss.backward()
                optimizer.step()

            # evaluate
            mae = self.evaluate(test_data)
            if mae < min_mae:
                min_mae = mae
            print('Epoch: {}, MAE: {:.4f}, Min MAE: {:.4f}'.format(epoch, mae, min_mae))

        # save predicted density map and segmentation map
        self.evaluate(test_data, save_ce=True)
        res = {'mae': min_mae}
        return res

    def evaluate(self, data, save_ce=False):
        if save_ce:
            if not os.path.exists('./res_dub'):
                os.mkdir('./res_dub')

            if not os.path.exists('./res_dub/uncer_csr'):
                os.mkdir('./res_dub/uncer_csr')

        self.clf.eval()
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        mae_sum = 0
        import csv
        header = ['density', 'ground_truth', 'uncertainty']
        rows = []
        with torch.no_grad():
            for batch_idx, (x, y, idxs) in enumerate(loader):
                x, y = x.to(self.device).float(), y.to(self.device)
                y = torch.nn.functional.interpolate(y.unsqueeze(0), size=(y.shape[1] // 16, y.shape[2] // 16),
                                                    mode='bilinear', align_corners=True) * 16 * 16
                density, uncertainty = self.clf(x)
                mae = abs(density.sum() - y.sum())
                mae_sum += mae.item()
                if save_ce:
                    uncertainty = uncertainty.squeeze(0).squeeze(0).data.cpu().numpy()
                    uncertainty_map = (uncertainty * 255).astype(np.uint8)
                    cv2.imwrite('res_dub/uncer_csr/uncer_{}.png'.format(idxs.item()), uncertainty_map)

                # Save density map count
                rows.append([density.sum().item(), y.sum().item(), uncertainty.mean().item()])
        mae_sum /= len(loader)

        if save_ce:
            with open(f'res_dub/res_uncer_csr.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(rows)
        return mae_sum

    def predict(self, data):
        self.clf.eval()
        preds = []
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.clf(x)
                preds.append(out.data.cpu().numpy())
        return preds

    def predict_single(self, data):
        self.clf.eval()
        data = data.to(self.device)
        return self.clf(data).data.cpu()


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class DUBNet(nn.Module):
    def __init__(self, block=Bottleneck, num_blocks=None, load_weights=False, num_heads=10):
        super(DUBNet, self).__init__()
        # Resnet 50
        if num_blocks is None:
            num_blocks = [3, 4, 6]
        self.in_planes = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.frontend = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            self._make_layer(block, 64, num_blocks[0], stride=1),
            self._make_layer(block, 128, num_blocks[1], stride=2),
            self._make_layer(block, 256, num_blocks[2], stride=2)
        )

        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.backend = make_layers(self.backend_feat, in_channels=1024, dilation=True)
        self.heads = nn.ModuleList([nn.Conv2d(64, 1, kernel_size=1) for _ in range(num_heads)])
        self.logvar = nn.Conv2d(64, 1, kernel_size=1)

        # Init weights
        if not load_weights:
            mod = models.resnet50(pretrained=True)
            self._initialize_weights()
            items = list(self.frontend.state_dict().items())
            _items = list(mod.state_dict().items())
            for i in range(len(self.frontend.state_dict().items())):
                if len(items[i][1].shape) > 0:  # 跳过0维参数
                    items[i][1].data[:] = _items[i][1].data[:]

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Forward for Resnet 50
        x = self.frontend(x)
        x = self.backend(x)

        if self.training:
            head = random.choice(self.heads)
            density = head(x)
        else:
            # In evaluation mode, return the average output of all heads
            density = torch.mean(torch.stack([head(x) for head in self.heads]), dim=0)

        logvar = self.logvar(x)
        density = F.relu(density)
        logvar = F.softplus(logvar)

        # density = F.interpolate(density, scale_factor=16, mode='bilinear', align_corners=False)
        # logvar = F.interpolate(logvar, scale_factor=16, mode='bilinear', align_corners=False)
        return density, logvar


class CSR_DUBNet(nn.Module):
    def __init__(self, load_weights=False, num_heads=10):
        super(CSR_DUBNet, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.heads = nn.ModuleList([nn.Conv2d(64, 1, kernel_size=1) for _ in range(num_heads)])
        self.logvar = nn.Conv2d(64, 1, kernel_size=1)

        if not load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            items = list(self.frontend.state_dict().items())
            _items = list(mod.state_dict().items())
            for i in range(len(self.frontend.state_dict().items())):
                items[i][1].data[:] = _items[i][1].data[:]

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)

        if self.training:
            head = random.choice(self.heads)
            density = head(x)
        else:
            # In evaluation mode, return the average output of all heads
            density = torch.mean(torch.stack([head(x) for head in self.heads]), dim=0)

        logvar = self.logvar(x)
        density = F.relu(density)
        logvar = F.relu(logvar)

        density = F.interpolate(density, scale_factor=8, mode='bilinear', align_corners=False)
        logvar = F.interpolate(logvar, scale_factor=8, mode='bilinear', align_corners=False)
        return density, logvar

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
