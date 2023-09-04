import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from torch.autograd import Variable
import torch.optim.lr_scheduler as lr_scheduler

from copy import deepcopy
from tqdm import tqdm

from utils_crowd.custom_collate import pssw_resize_collate_fn

import sys


# LossPredictionLoss
def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape

    input = (input - input.flip(0))[:len(input) // 2]  # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], batch size = 2B
    target = (target - target.flip(0))[:len(target) // 2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1  # 1 operation which is defined by the authors

    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0)  # Note that the size of input is already haved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        raise NotImplementedError()

    return loss


def LossPredLoss_Rank(input, target, margin=1.0, reduction='mean'):
    # This is another implementation of LossPredLoss which makes the training objective simpler.
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    criterion = nn.BCELoss()
    input = (input - input.flip(0))[
            :len(input) // 2]  # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target) // 2]
    target = target.detach()
    diff = torch.sigmoid(input)
    one = torch.sign(torch.clamp(target, min=0))  # 1 operation which is defined by the authors

    if reduction == 'mean':
        loss = criterion(diff, one)
    elif reduction == 'none':
        loss = criterion(diff, one)
    else:
        NotImplementedError()

    return loss


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Net_LPL:
    def __init__(self, net, params, device, net_lpl):
        self.net = net
        self.params = params
        self.device = device
        self.net_lpl = net_lpl

    def train(self, data, test_data, weight=1.0, margin=1.0, lpl_epoch=120):
        n_epoch = self.params['n_epoch']
        epoch_loss = lpl_epoch

        dim = data.X.shape[1:]
        self.clf = self.net(dim=dim, pretrained=self.params['pretrained'], num_classes=self.params['num_class'])

        if torch.cuda.device_count() > 1:
            self.clf = nn.DataParallel(self.clf)
            self.net_lpl = nn.DataParallel(self.net_lpl)
        self.clf.to(self.device)
        self.net_lpl.to(self.device)

        if self.params['optimizer'] == 'Adam':
            optimizer = optim.Adam(self.clf.parameters(), **self.params['optimizer_args'])
            optimizer_lpl = optim.Adam(self.net_lpl.parameters(), **self.params['optimizer_args'])
        elif self.params['optimizer'] == 'SGD':
            optimizer = optim.SGD(self.clf.parameters(), **self.params['optimizer_args'])
            optimizer_lpl = optim.SGD(self.net_lpl.parameters(), **self.params['optimizer_args'])
        else:
            raise NotImplementedError

        sched_backbone = lr_scheduler.MultiStepLR(optimizer, milestones=[160])
        sched_lpl = lr_scheduler.MultiStepLR(optimizer_lpl, milestones=[160])

        loader = DataLoader(data, shuffle=True, **self.params['loader_tr_args'])
        self.clf.train()
        self.net_lpl.train()
        for epoch in tqdm(range(1, n_epoch + 1), ncols=100):
            sched_backbone.step()
            sched_lpl.step()

            for batch_idx, (x, y, idxs) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                optimizer_lpl.zero_grad()
                out, feature = self.clf(x)
                cross_ent = nn.CrossEntropyLoss(reduction='none')
                target_loss = cross_ent(out, y)
                if epoch >= epoch_loss:
                    feature[0] = feature[0].detach()
                    feature[1] = feature[1].detach()
                    feature[2] = feature[2].detach()
                    feature[3] = feature[3].detach()

                pred_loss = self.net_lpl(feature)
                pred_loss = pred_loss.view(pred_loss.size(0))

                backbone_loss = torch.sum(target_loss) / target_loss.size(0)
                module_loss = LossPredLoss(pred_loss, target_loss, margin)
                loss = backbone_loss + weight * module_loss
                loss.backward()
                optimizer.step()
                optimizer_lpl.step()

    def predict(self, data):
        self.clf.eval()
        preds = torch.zeros(len(data), dtype=data.Y.dtype)
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                pred = out.max(1)[1]
                preds[idxs] = pred.cpu()
        return preds

    def predict_prob(self, data):
        self.clf.eval()
        probs = torch.zeros([len(data), len(np.unique(data.Y))])
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu()
        return probs

    def predict_prob_dropout(self, data, n_drop=10):
        self.clf.train()
        probs = torch.zeros([len(data), len(np.unique(data.Y))])
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        for i in range(n_drop):
            with torch.no_grad():
                for x, y, idxs in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[idxs] += prob.cpu()
        probs /= n_drop
        return probs

    def predict_prob_dropout_split(self, data, n_drop=10):
        self.clf.train()
        probs = torch.zeros([n_drop, len(data), len(np.unique(data.Y))])
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        for i in range(n_drop):
            with torch.no_grad():
                for x, y, idxs in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[i][idxs] += F.softmax(out, dim=1).cpu()
        return probs

    def get_model(self):
        return self.clf

    def get_embeddings(self, data):
        self.clf.eval()
        embeddings = torch.zeros([len(data), self.clf.get_embedding_dim()])
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                embeddings[idxs] = e1.cpu()
        return embeddings

    def get_grad_embeddings(self, data):
        self.clf.eval()
        embDim = self.clf.get_embedding_dim()
        nLab = self.params['num_class']
        embeddings = np.zeros([len(data), embDim * nLab])

        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = Variable(x.to(self.device)), Variable(y.to(self.device))
                cout, out = self.clf(x)
                out = out.data.cpu().numpy()
                batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()
                maxInds = np.argmax(batchProbs, 1)
                for j in range(len(y)):
                    for c in range(nLab):
                        if c == maxInds[j]:
                            embeddings[idxs[j]][embDim * c: embDim * (c + 1)] = deepcopy(out[j]) * (
                                        1 - batchProbs[j][c]) * -1.0
                        else:
                            embeddings[idxs[j]][embDim * c: embDim * (c + 1)] = deepcopy(out[j]) * (
                                        -1 * batchProbs[j][c]) * -1.0

        return embeddings


class Net_LPL_CSR:
    def __init__(self, net, params, device, net_lpl, TA_VAAL=False):
        self.net = net
        self.params = params
        self.device = device
        self.net_lpl = net_lpl
        self.TA_VAAL = TA_VAAL

    def train(self, data, test_data, weight=1.0, margin=1.0, lpl_epoch=120):
        n_epoch = self.params['n_epoch']
        epoch_loss = lpl_epoch

        self.clf = self.net()

        if torch.cuda.device_count() > 1:
            self.clf = nn.DataParallel(self.clf)
            self.net_lpl = nn.DataParallel(self.net_lpl)
        self.clf.to(self.device)
        self.net_lpl.to(self.device)

        if self.params['optimizer'] == 'Adam':
            optimizer = optim.Adam(self.clf.parameters(), **self.params['optimizer_args'])
            optimizer_lpl = optim.Adam(self.net_lpl.parameters(), **self.params['optimizer_args'])
        elif self.params['optimizer'] == 'SGD':
            optimizer = optim.SGD(self.clf.parameters(), **self.params['optimizer_args'])
            optimizer_lpl = optim.SGD(self.net_lpl.parameters(), **self.params['optimizer_args'])
        else:
            raise NotImplementedError

        # sched_backbone = lr_scheduler.MultiStepLR(optimizer, milestones=[160])
        # sched_lpl = lr_scheduler.MultiStepLR(optimizer_lpl, milestones=[160])

        if self.params['loader_tr_args']['batch_size'] == 1:
            # Assign it to 2 to avoid error
            self.params['loader_tr_args']['batch_size'] = 2

        loader = DataLoader(data, shuffle=True, collate_fn=pssw_resize_collate_fn, **self.params['loader_tr_args'])
        self.clf.train()
        self.net_lpl.train()
        min_mae = sys.maxsize
        min_mse = sys.maxsize
        min_density = None
        min_density_mse = None
        criterion = nn.MSELoss(reduction='none')
        for epoch in tqdm(range(1, n_epoch + 1), ncols=100):
            # sched_backbone.step()
            # sched_lpl.step()

            for batch_idx, (x, y, idxs) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                optimizer_lpl.zero_grad()
                out, feature = self.clf(x)
                out = out.squeeze(1)
                target_loss = criterion(out, y)
                target_loss = torch.sum(target_loss, dim=(1, 2))
                if epoch >= epoch_loss:
                    feature[0] = feature[0].detach()
                    feature[1] = feature[1].detach()
                    feature[2] = feature[2].detach()
                    feature[3] = feature[3].detach()

                pred_loss = self.net_lpl(feature)
                pred_loss = pred_loss.view(pred_loss.size(0))

                backbone_loss = torch.sum(target_loss) / target_loss.size(0)

                # Two ways to calculate the module loss
                if not self.TA_VAAL:
                    module_loss = LossPredLoss(pred_loss, target_loss, margin)
                else:
                    module_loss = LossPredLoss_Rank(pred_loss, target_loss, margin)
                loss = backbone_loss + weight * module_loss
                loss.backward()
                optimizer.step()
                optimizer_lpl.step()

                # evaluate
                mae, mse, density, density_mse = self.evaluate(test_data)
                if mae < min_mae:
                    min_mae = mae
                    min_mse = mse
                    min_density = density
                    min_density_mse = density_mse
                    # torch.save(self.clf.state_dict(), './ckpt/best_model_lpl.pth')
                print('Epoch: {}, MAE: {:.4f}, Min MAE: {:.4f}'.format(epoch, mae, min_mae))

        # self.clf.load_state_dict(torch.load('./ckpt/best_model_lpl.pth'))
        res = {'mae': min_mae, 'mse': min_mse, 'density': min_density, 'density_mse': min_density_mse}
        return res

    def predict(self, data):
        self.clf.eval()
        preds = []
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, _ = self.clf(x)
                preds.append(out.data.cpu().numpy())
        return preds

    def evaluate(self, data):
        self.clf.eval()
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        loss = 0
        mse = 0
        density_eval = {'low': [], 'medium': [], 'high': []}
        density_eval_mse = {'low': [], 'medium': [], 'high': []}
        with torch.no_grad():
            for batch_idx, (x, y, idxs) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)
                out, _ = self.clf(x)
                mae = abs(out.sum() - y.sum())
                mse += (out.sum() - y.sum()) ** 2
                loss += mae.item()
                if y.sum() <= 50:
                    density_eval['low'].append(mae)
                    density_eval_mse['low'].append(mse)
                elif y.sum() <= 500:
                    density_eval['medium'].append(mae)
                    density_eval_mse['medium'].append(mse)
                else:
                    density_eval['high'].append(mae)
                    density_eval_mse['high'].append(mse)

        if density_eval is not None:
            for key in density_eval.keys():
                if density_eval[key]:
                    density_eval[key] = torch.mean(torch.stack(density_eval[key])).item()

                if density_eval_mse[key]:
                    density_eval_mse[key] = torch.mean(torch.stack(density_eval_mse[key]))
                    density_eval_mse[key] = math.sqrt(density_eval_mse[key].item())

        loss /= len(loader)
        mse /= len(loader)
        mse = mse ** 0.5
        return loss, mse, density_eval, density_eval_mse

    def get_model(self):
        return self.clf


# class Net_Crowd_Counting_SGANet(Net):
#     def __init__(self, net, params, device):
#         super(Net_Crowd_Counting_SGANet, self).__init__(net, params, device)
#
#     def train(self, labelled_data, test_data, all_data=None, labeled_idx=None):
#         n_epoch = self.params['n_epoch']
#         self.clf = self.net()
#         model_urls = {
#             # Inception v3 ported from TensorFlow
#             'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
#         }
#         self.clf.load_state_dict(model_zoo.load_url(model_urls['inception_v3_google']), strict=False)
#
#         if torch.cuda.device_count() > 1:
#             self.clf = nn.DataParallel(self.clf)
#         self.clf.to(self.device)
#
#         self.clf.train()
#         if self.params['optimizer'] == 'Adam':
#             optimizer = optim.Adam(self.clf.parameters(), **self.params['optimizer_args'])
#             lr_schedule = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
#         elif self.params['optimizer'] == 'SGD':
#             optimizer = optim.SGD(self.clf.parameters(), **self.params['optimizer_args'])
#             lr_schedule = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)
#         else:
#             raise NotImplementedError
#
#         cl_loss = True
#         seg_loss = True
#         loader_labelled = DataLoader(labelled_data, shuffle=True, **self.params['loader_tr_args'])
#
#         # In crowd counting the batch size is one, so we don't need to divide by it
#         criterion1 = nn.MSELoss(reduce=False)  # for density map loss
#         criterion2 = nn.BCELoss()  # for segmentation map loss
#         best_model_wts = copy.deepcopy(self.clf.state_dict())
#         best_mae_val = 1e6
#         best_mse_val = 1e6
#         best_mae_by_val = 1e6
#         best_mae_by_test = 1e6
#         best_mse_by_val = 1e6
#         best_mse_by_test = 1e6
#         min_density_eval = None
#         min_density_eval_mse = None
#         for epoch in tqdm(range(1, n_epoch + 1), ncols=100):
#             self.clf.train()  # Set model to training mode
#             running_loss = 0.0
#             for index, (inputs, labels, fbs, idx) in enumerate(loader_labelled):
#                 labels = labels * 100
#                 labels = skimage.measure.block_reduce(labels.cpu().numpy(), (1, 1, 1, 4, 4), np.sum)
#                 fbs = skimage.measure.block_reduce(fbs.cpu().numpy(), (1, 1, 1, 4, 4), np.max)
#                 fbs = np.float32(fbs > 0)
#                 labels = torch.from_numpy(labels)
#                 fbs = torch.from_numpy(fbs)
#                 labels = labels.to(self.device)
#                 fbs = fbs.to(self.device)
#                 inputs = inputs.to(self.device)
#                 inputs = inputs.view(-1, inputs.shape[2], inputs.shape[3], inputs.shape[4])
#                 labels = labels.view(-1, labels.shape[3], labels.shape[4])
#                 fbs = fbs.view(-1, fbs.shape[3], fbs.shape[4])
#                 inputs = inputs.float()
#                 labels = labels.float()
#                 # zero the parameter gradients
#                 optimizer.zero_grad()
#
#                 # forward
#                 # track history if only in train
#                 with torch.set_grad_enabled(True):
#                     output, fbs_out = self.clf(inputs)
#                     loss_den = criterion1(output, labels)
#                     loss_seg = criterion2(fbs_out, fbs)
#                     if cl_loss:
#                         th = 0.1 * epoch + 5  # cl2
#                     else:
#                         th = 1000  # no curriculum loss when th is set a big number
#                     weights = th / (F.relu(labels - th) + th)
#                     loss_den = loss_den * weights
#                     loss_den = loss_den.sum() / weights.sum()
#                     if seg_loss:
#                         loss = loss_den + 20 * loss_seg
#                     else:
#                         loss = loss_den
#
#                     loss.backward()
#                     optimizer.step()
#                 running_loss += loss.item() * inputs.size(0)
#             epoch_loss = running_loss / len(labelled_data)
#             if self.params['optimizer'] == 'Adam':
#                 lr_schedule.step()
#             mae, _, mse, density_eval, density_eval_mse = self.evaluate(test_data)
#             print('Epoch: {}/{}, Loss: {:.4f}, MAE: {:.4f}, MSE: {:.4f}, Min MAE: {:.4f}'.format(epoch, n_epoch, epoch_loss,
#                                                                                  mae, mse, best_mae_val))
#             if mae < best_mae_val:
#                 best_mae_val = mae
#                 best_mse_val = mse
#                 min_density_eval = density_eval
#                 min_density_eval_mse = density_eval_mse
#
#                 # 这里还没写完
#         res = {'mae': best_mae_val, 'mse': best_mse_val, 'density_eval': min_density_eval, 'density_eval_mse': min_density_eval_mse}
#         return res
#
#     def evaluate(self, data):
#
#         self.clf.eval()
#         loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
#         mae_sum = 0
#         mse_sum = 0
#         ce_sum = 0
#         # ce_loss = nn.CrossEntropyLoss(reduction='sum')
#         import csv
#         header = ['idx', 'density', 'ground_truth']
#         rows = []
#         density_eval = {'low': [], 'medium': [], 'high': []}
#         density_eval_mse = {'low': [], 'medium': [], 'high': []}
#         with torch.no_grad():
#             for batch_idx, (x, y, idxs) in enumerate(loader):
#                 x, y = x.to(self.device).float(), y.to(self.device)
#                 x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])
#                 y = y.view(-1, y.shape[3], y.shape[4])
#                 # y = torch.nn.functional.interpolate(y.unsqueeze(0), size=(y.shape[1] // 8, y.shape[2] // 8),
#                 #                                     mode='bilinear', align_corners=True).squeeze(0) * 64
#                 # Try to avoid the out of memory error, using sliding window to predict
#                 if self.params['name'] == 'JHUCrowd' or self.params['name'] == 'UCF-QNRF':
#                     b, c, h, w = x.shape
#                     input_list = []
#                     if h >= 3584 or w >= 3584:
#                         h_stride = int(math.ceil(1.0 * h / 3584))
#                         w_stride = int(math.ceil(1.0 * w / 3584))
#                         h_step = h // h_stride
#                         w_step = w // w_stride
#                         for i in range(h_stride):
#                             for j in range(w_stride):
#                                 h_start = i * h_step
#                                 if i != h_stride - 1:
#                                     h_end = (i + 1) * h_step
#                                 else:
#                                     h_end = h
#                                 w_start = j * w_step
#                                 if j != w_stride - 1:
#                                     w_end = (j + 1) * w_step
#                                 else:
#                                     w_end = w
#                                 input_list.append(x[:, :, h_start:h_end, w_start:w_end])
#                         with torch.no_grad():
#                             density_sum = 0.0
#                             for idx, input in enumerate(input_list):
#                                 density, seg = self.clf(input)
#                                 density = density / 100
#                                 density_sum += density.sum()
#                     else:
#                         density, seg  = self.clf(x)
#                         density = density / 100
#                         density_sum = density.sum()
#                 else:
#                     density, seg = self.clf(x)
#                     density = density / 100
#                     density_sum = density.sum()
#
#                 density = density.squeeze(0)
#                 density = density / 100
#                 seg_gt_raw = torch.where(y > 0, torch.tensor(1), torch.tensor(0))
#                 # seg_gt = F.one_hot(seg_gt_raw, num_classes=2).permute(0, 3, 1, 2).float()
#                 mae = abs(density_sum - y.sum())
#                 mse = (density_sum - y.sum()) ** 2
#                 mse_sum += mse.item()
#                 mae_sum += mae.item()
#                 # ce_sum += ce_loss(seg, seg_gt).item()
#                 if y.sum() <= 50:
#                     density_eval['low'].append(mae)
#                     density_eval_mse['low'].append(mse)
#                 elif y.sum() <= 500:
#                     density_eval['medium'].append(mae)
#                     density_eval_mse['medium'].append(mse)
#                 else:
#                     density_eval['high'].append(mae)
#                     density_eval_mse['high'].append(mse)
#
#         mae_sum /= len(loader)
#         ce_sum /= len(loader)
#         mse_sum /= len(loader)
#         mse_sum = math.sqrt(mse_sum)
#         if density_eval is not None:
#             for key in density_eval.keys():
#                 if density_eval[key]:
#                     density_eval[key] = torch.mean(torch.stack(density_eval[key])).item()
#
#                 if density_eval_mse[key]:
#                     density_eval_mse[key] = torch.mean(torch.stack(density_eval_mse[key]))
#                     density_eval_mse[key] = math.sqrt(density_eval_mse[key].item())
#
#         return mae_sum, ce_sum, mse_sum, density_eval, density_eval_mse
#
#     def get_seg_density_predictions(self, data):
#         data = data.squeeze(0).float()
#         density, seg = self.clf(data)
#         # TODO
#         density_original = density / 100
#         seg = torch.log(seg / (1 - seg))
#         return density_original, seg
#
#     def predict_single(self, data):
#         density, _ = self.clf(data)
#         return density / 100


class MNIST_Net_LPL(nn.Module):
    def __init__(self, dim=28 * 28, pretrained=False, num_classes=10):
        super().__init__()
        resnet18 = models.resnet18(pretrained=pretrained)
        self.features = nn.Sequential(*list(resnet18.children())[:-1])

        self.feature0 = nn.Sequential(*list(resnet18.children())[0:3])
        self.feature1 = nn.Sequential(*list(resnet18.children())[4])
        self.feature2 = nn.Sequential(*list(resnet18.children())[5])
        self.feature3 = nn.Sequential(*list(resnet18.children())[6])
        self.feature4 = nn.Sequential(*list(resnet18.children())[7])
        self.feature5 = nn.Sequential(*list(resnet18.children())[8:9])
        self.conv = nn.Conv2d(1, 3, kernel_size=1)
        self.classifier = nn.Linear(resnet18.fc.in_features, num_classes)
        self.dim = resnet18.fc.in_features

    def forward(self, x):
        x = self.conv(x)
        x0 = self.feature0(x)
        x1 = self.feature1(x0)
        x2 = self.feature2(x1)
        x3 = self.feature3(x2)
        x4 = self.feature4(x3)
        x5 = self.feature5(x4)
        output = x5.view(x5.size(0), -1)
        output = self.classifier(output)
        return output, [x1, x2, x3, x4]

    def get_embedding_dim(self):
        return self.dim


class CIFAR10_Net_LPL(nn.Module):
    def __init__(self, dim=28 * 28, pretrained=False, block=BasicBlock, num_blocks=None, num_classes=10):
        super(CIFAR10_Net_LPL, self).__init__()
        if num_blocks is None:
            num_blocks = [2, 2, 2, 2]
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out = F.avg_pool2d(out4, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, [out1, out2, out3, out4]

    def get_embedding_dim(self):
        return self.dim


class openml_Net(nn.Module):
    def __init__(self, dim=28 * 28, embSize=256, pretrained=False, num_classes=10):
        super(openml_Net, self).__init__()
        self.embSize = embSize
        self.dim = int(np.prod(dim))
        self.lm1 = nn.Linear(self.dim, embSize)
        self.lm2 = nn.Linear(embSize, num_classes)

    def forward(self, x):
        x = x.view(-1, self.dim)
        emb = F.relu(self.lm1(x))
        out = self.lm2(emb)
        return out, emb

    def get_embedding_dim(self):
        return self.embSize


class PneumoniaMNIST_Net_LPL(nn.Module):
    def __init__(self, dim=28 * 28, pretrained=False, num_classes=10):
        super().__init__()
        resnet18 = models.resnet18(pretrained=pretrained)
        self.features = nn.Sequential(*list(resnet18.children())[:-1])
        self.feature0 = nn.Sequential(*list(resnet18.children())[0:3])
        self.feature1 = nn.Sequential(*list(resnet18.children())[4])
        self.feature2 = nn.Sequential(*list(resnet18.children())[5])
        self.feature3 = nn.Sequential(*list(resnet18.children())[6])
        self.feature4 = nn.Sequential(*list(resnet18.children())[7])
        self.feature5 = nn.Sequential(*list(resnet18.children())[8:9])
        self.classifier = nn.Linear(512, num_classes)

        self.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.feature0[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.dim = resnet18.fc.in_features

    def forward(self, x):
        x0 = self.feature0(x)
        x1 = self.feature1(x0)
        x2 = self.feature2(x1)
        x3 = self.feature3(x2)
        x4 = self.feature4(x3)
        x5 = self.feature5(x4)
        output = x5.view(x5.size(0), -1)
        output = self.classifier(output)
        return output, [x1, x2, x3, x4]

    def get_embedding_dim(self):
        return self.dim


class waterbirds_Net_LPL(nn.Module):
    def __init__(self, dim=28 * 28, pretrained=False, num_classes=10):
        super().__init__()
        resnet18 = models.resnet18(pretrained=pretrained)
        self.features = nn.Sequential(*list(resnet18.children())[:-1])
        self.feature0 = nn.Sequential(*list(resnet18.children())[0:3])
        self.feature1 = nn.Sequential(*list(resnet18.children())[4])
        self.feature2 = nn.Sequential(*list(resnet18.children())[5])
        self.feature3 = nn.Sequential(*list(resnet18.children())[6])
        self.feature4 = nn.Sequential(*list(resnet18.children())[7])
        self.feature5 = nn.Sequential(*list(resnet18.children())[8:9])
        self.classifier = nn.Linear(resnet18.fc.in_features, num_classes)
        self.dim = resnet18.fc.in_features

    def forward(self, x):
        x0 = self.feature0(x)
        x1 = self.feature1(x0)
        x2 = self.feature2(x1)
        x3 = self.feature3(x2)
        x4 = self.feature4(x3)
        x5 = self.feature5(x4)
        output = x5.view(x5.size(0), -1)
        output = self.classifier(output)
        return output, [x1, x2, x3, x4]

    def get_embedding_dim(self):
        return self.dim


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


class CSR_Net_LPL(nn.Module):
    def __init__(self, load_weights=False):
        super(CSR_Net_LPL, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.feature_pos = [3, 8, 15, 22]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        if not load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            _items = list(mod.state_dict().items())
            items = list(self.frontend.state_dict().items())
            for i in range(len(self.frontend.state_dict().items())):
                items[i][1].data[:] = _items[i][1].data[:]

    def forward(self, x):
        feature_maps = []
        for i in range(len(self.frontend)):
            x = self.frontend[i](x)
            if i in self.feature_pos:
                feature_maps.append(x)

        x = self.backend(x)
        x = self.output_layer(x)
        x = nn.functional.interpolate(x, scale_factor=8)
        return x, feature_maps

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def get_lossnet(name):
    if name == 'PneumoniaMNIST':
        return LossNet(feature_sizes=[224, 112, 56, 28], num_channels=[64, 128, 256, 512], interm_dim=128)
    elif 'MNIST' in name:
        return LossNet(feature_sizes=[14, 7, 4, 2], num_channels=[64, 128, 256, 512], interm_dim=128)
    elif 'CIFAR' in name:
        return LossNet(feature_sizes=[32, 16, 8, 4], num_channels=[64, 128, 256, 512], interm_dim=128)
    elif 'ImageNet' in name:
        return LossNet(feature_sizes=[64, 32, 16, 8], num_channels=[64, 128, 256, 512], interm_dim=128)
    elif 'BreakHis' in name:
        return LossNet(feature_sizes=[224, 112, 56, 28], num_channels=[64, 128, 256, 512], interm_dim=128)
    elif 'waterbirds' in name:
        return LossNet(feature_sizes=[128, 64, 32, 16], num_channels=[64, 128, 256, 512], interm_dim=128)
    elif 'shanghaitechA' or 'shangtaitechB' in name:
        # Using global average pooling, so feature sizes are not needed
        return LossNet(num_channels=[64, 128, 256, 512], interm_dim=128)
    else:
        raise NotImplementedError


class LossNet(nn.Module):
    def __init__(self, feature_sizes=[28, 14, 7, 4], num_channels=[64, 128, 256, 512], interm_dim=128):
        super(LossNet, self).__init__()

        self.GAP1 = nn.AdaptiveAvgPool2d((1, 1))
        self.GAP2 = nn.AdaptiveAvgPool2d((1, 1))
        self.GAP3 = nn.AdaptiveAvgPool2d((1, 1))
        self.GAP4 = nn.AdaptiveAvgPool2d((1, 1))

        self.FC1 = nn.Linear(num_channels[0], interm_dim)
        self.FC2 = nn.Linear(num_channels[1], interm_dim)
        self.FC3 = nn.Linear(num_channels[2], interm_dim)
        self.FC4 = nn.Linear(num_channels[3], interm_dim)

        self.linear = nn.Linear(4 * interm_dim, 1)

    def forward(self, features):
        out1 = self.GAP1(features[0])
        out1 = out1.view(out1.size(0), -1)
        out1 = F.relu(self.FC1(out1))

        out2 = self.GAP2(features[1])
        out2 = out2.view(out2.size(0), -1)
        out2 = F.relu(self.FC2(out2))

        out3 = self.GAP3(features[2])
        out3 = out3.view(out3.size(0), -1)
        out3 = F.relu(self.FC3(out3))

        out4 = self.GAP4(features[3])
        out4 = out4.view(out4.size(0), -1)
        out4 = F.relu(self.FC4(out4))

        out = self.linear(torch.cat((out1, out2, out3, out4), 1))
        return out


class SGANet_LPL(nn.Module):
    """
    Crowd Counting via Segmentation Guided Attention Networks and Curriculum Loss

    code directly taken from https://github.com/hellowangqian/sganet-crowd-counting with little modifications.
    """
    def __init__(self, num_classes=1000, aux_logits=False, transform_input=False):
        super(SGANet_LPL, self).__init__()
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3, padding=1)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3, padding=1)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)
        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes)
        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)

        self.relu = nn.ReLU(inplace=True)
        self.sigm = nn.Sigmoid()
        self.lconv1 = nn.Conv2d(288, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.lconv2 = nn.Conv2d(768, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.lconv3 = nn.Conv2d(2048, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.att_conv = nn.Conv2d(2048, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.Tensor(X.rvs(m.weight.numel()))
                values = values.view(m.weight.size())
                m.weight.data.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        # x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        # x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288
        # 128x128x288
        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 64x64x768
        # 17 x 17 x 768

        if self.training and self.aux_logits:
            aux = self.AuxLogits(x)
        # 17 x 17 x 768
        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.upsample(x)
        attention_map = self.sigm(self.att_conv(x))
        feature_map3 = self.Mixed_7c(x)
        feature_map4 = feature_map3 * attention_map
        # 32x32x2048
        # feature_map4 = F.avg_pool2d(feature_map3,2)
        # x_cat = feature_map3
        # density_map1 = self.lconv1(feature_map1)
        # density_map1 = density_map1.view(-1,density_map1.size(2),density_map1.size(3))
        # density_map2 = self.lconv2(feature_map2)
        # density_map2 = density_map2.view(-1,density_map2.size(2),density_map2.size(3))
        density_map3 = self.lconv3(feature_map4)
        density_map3 = self.relu(density_map3)
        density_map3 = density_map3.view(-1, density_map3.size(2), density_map3.size(3))
        attention_map = attention_map.view(-1, attention_map.size(2), attention_map.size(3))

        ## TODO
        # with torch.no_grad():
        #     density_map_original = self.lconv3(feature_map3)
        #     density_map_original = self.relu(density_map_original)
        #     density_map_original = density_map_original.view(-1, density_map_original.size(2), density_map_original.size(3))
        # density_map4 = self.lconv4(feature_map4)
        # density_map4 = density_map4.view(-1,density_map4.size(2),density_map4.size(3))
        # density_map = F.avg_pool2d(density_map,kernel_size=2)
        return density_map3, attention_map


class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.contiguous().view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        x = x.permute(1, 0, 2)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2, padding=1)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):

    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2, padding=1)

        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv2d(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        # 17 x 17 x 768
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # 5 x 5 x 768
        x = self.conv0(x)
        # 5 x 5 x 128
        x = self.conv1(x)
        # 1 x 1 x 768
        x = x.view(x.size(0), -1)
        # 768
        x = self.fc(x)
        # 1000
        return x


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)



if __name__ == "__main__":
    a = LossNet()
    # Print number of parameters
    print(sum(p.numel() for p in a.parameters() if p.requires_grad))