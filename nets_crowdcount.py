import collections
import re

import numpy as np
import skimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.autograd import Function
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
import torch.utils.model_zoo as model_zoo
import torch.distributed as dist
import torchvision
import torchvision.models as models
from tqdm import tqdm
import sys
from utils_crowd.custom_collate import pssw_collate_fn, pssw_resize_collate_fn, SHHA_collate
from nets import Net
import cv2
import gc
import os
import math
import random
import copy

class Net_Crowd_Counting_Project(Net):
    def __init__(self, net, params, device):
        super(Net_Crowd_Counting_Project, self).__init__(net, params, device)

    def train(self, labelled_data, test_data):
        n_epoch = self.params['n_epoch']
        self.clf = self.net()
        self.dis = False
        if torch.cuda.device_count() > 1:
            # TODO, not implemented yet
            dis, rank = init_distributed_mode()
            self.dis = dis
            self.rank = rank
            if dis:
                self.clf = nn.parallel.DistributedDataParallel(self.clf, device_ids=[rank])
                print('Using DistributedDataParallel')

        self.clf.to(self.device)

        self.clf.train()

        # torch.save(self.clf.state_dict(), './ckpt/init_model.pt')
        if self.params['optimizer'] == 'Adam':
            optimizer = optim.Adam(self.clf.parameters(),  **self.params['optimizer_args'])
            lr_schedule = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=1)
        elif self.params['optimizer'] == 'SGD':
            optimizer = optim.SGD(self.clf.parameters(), **self.params['optimizer_args'])
            lr_schedule = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)
        else:
            raise NotImplementedError

        if self.params['name'] == 'UCF-QNRF' or self.params['name'] == 'JHUCrowd' or self.params['name'] == 'IOCfish':
            loader_labelled = DataLoader(labelled_data, shuffle=True, collate_fn=SHHA_collate, **self.params['loader_tr_args'])
        else:
            loader_labelled = DataLoader(labelled_data, shuffle=True, **self.params['loader_tr_args'])

        min_mae = sys.maxsize
        # In crowd counting the batch size is one, so we don't need to divide by it

        if self.params['name'] == 'UCF-QNRF' or self.params['name'] == 'IOCfish':
            criterion = nn.MSELoss(reduction='mean')
            seg_weight = 1.
            label_expansion = 100.
        else:
            criterion = nn.MSELoss(reduction='sum')
            seg_weight = 1.
            label_expansion = 1.

        bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
        best_epoch = 0
        min_mse_sum = 0
        min_density_eval = None
        min_density_eval_mse = None

        # for debugging
        fail_to_train = False
        count_fail = 0
        for epoch in tqdm(range(1, n_epoch + 1), ncols=100):
            if self.params['optimizer'] == 'Adam':
                lr_schedule.step()

            # Not implemented yet TODO
            if self.dis:
                datasampler = DistributedSampler(labelled_data, num_replicas=dist.get_world_size(), rank=self.rank)
                datasampler.set_epoch(epoch)
                # Reinitialize the data loader
                if self.params['name'] == 'UCF-QNRF' or self.params['name'] == 'JHUCrowd' or self.params[
                    'name'] == 'IOCfish':
                    loader_labelled = DataLoader(labelled_data, shuffle=True, collate_fn=SHHA_collate, sampler=datasampler,
                                                 **self.params['loader_tr_args'])
                else:
                    loader_labelled = DataLoader(labelled_data, shuffle=True, sampler=datasampler, **self.params['loader_tr_args'])

            # Training labelled data
            for batch_idx, (x, y, idxs) in enumerate(loader_labelled):
                x, y = x.to(self.device), y.to(self.device)
                y = y * label_expansion
                # y = torch.nn.functional.interpolate(y.unsqueeze(0), size=(y.shape[1] // 8, y.shape[2] // 8),
                #                                     mode='bilinear', align_corners=True).squeeze(0) * 64
                seg_gt = torch.where(y > 0, torch.tensor(1), torch.tensor(0)).float()
                # seg_gt = F.one_hot(seg_gt_raw, num_classes=2).permute(0, 3, 1, 2).float()
                optimizer.zero_grad()

                density, seg = self.clf(x)
                density = density.squeeze(1)
                seg = seg.squeeze(1)
                mse_loss = criterion(density, y)
                # nll1 = F.gaussian_nll_loss(density, y, uncertainty, reduction='mean')
                # Calculate density loss
                # nll = self.negative_log_likelihood(y, (density, uncertainty)).mean()

                # Calculate segmentation loss
                seg_loss = bce_loss(seg, seg_gt)
                # print('loss')
                # print(mse_loss.item(), seg_loss.item())
                # print('MSE: {:.4f}, Seg Loss: {:.4f}'.format(mse_loss.item(), 1e-2* seg_loss.item()))
                # Calculate total loss
                total_loss = mse_loss + seg_weight * seg_loss
                if fail_to_train:
                    return {'success': False}
                # if epoch % 50 == 0:
                #     print('MSE: {:.4f}, Seg Loss: {:.4f}, Total Loss: {:.4f}'.format(mse_loss.item(), seg_loss.item(), total_loss.item()))
                # print('MSE: {:.4f}, Seg Loss: {:.4f}, Total Loss: {:.4f}'.format(mse_loss.item(), seg_loss.item(), total_loss.item()))

                total_loss.backward()
                optimizer.step()

            # evaluate
            mae, mse_sum, density_eval, density_eval_mse = self.evaluate(test_data, label_expansion)
            # Train mae
            # train_mae, _, _, _, _ = self.evaluate(labelled_data, label_expansion)
            if self.params['name'] == 'shanghaitechB' and mae > 110:
                if count_fail < 10:
                    count_fail += 1
                else:
                    fail_to_train = True
            else:
                fail_to_train = False
            if mae < min_mae:
                min_mae = mae
                min_mse_sum = mse_sum
                min_density_eval = density_eval
                min_density_eval_mse = density_eval_mse
                # torch.save(self.clf.state_dict(), './ckpt/best_model_project.pth')
            # print('Epoch: {}, MAE: {:.4f}, Min MAE: {:.4f}, Train MAE:{:.4f}'.format(epoch, mae, min_mae, train_mae))
            print('Epoch: {}, MAE: {:.4f}, Min MAE: {:.4f}'.format(epoch, mae, min_mae))

        res = {'mae': min_mae,
               'mse': min_mse_sum,
               'density_eval': min_density_eval,
               'density_eval_mse': min_density_eval_mse,
               'success': True}
        return res

    def evaluate(self, data, label_expansion=1.):
        self.clf.eval()
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        mae_sum = 0
        mse_sum = 0

        density_eval = {'low': [], 'medium': [], 'high': []}
        density_eval_mse = {'low': [], 'medium': [], 'high': []}
        with torch.no_grad():
            for batch_idx, (x, y, idxs) in enumerate(loader):

                x, y = x.to(self.device).float(), y.to(self.device)
                # y = torch.nn.functional.interpolate(y.unsqueeze(0), size=(y.shape[1] // 8, y.shape[2] // 8),
                #                                     mode='bilinear', align_corners=True).squeeze(0) * 64
                # Try to avoid the out of memory error, using sliding window to predict
                if self.params['name'] == 'JHUCrowd' or self.params['name'] == 'UCF-QNRF':
                    b, c, h, w = x.shape
                    input_list = []
                    if h >= 3584 or w >= 3584:
                        h_stride = int(math.ceil(1.0 * h / 3584))
                        w_stride = int(math.ceil(1.0 * w / 3584))
                        h_step = h // h_stride
                        w_step = w // w_stride
                        for i in range(h_stride):
                            for j in range(w_stride):
                                h_start = i * h_step
                                if i != h_stride - 1:
                                    h_end = (i + 1) * h_step
                                else:
                                    h_end = h
                                w_start = j * w_step
                                if j != w_stride - 1:
                                    w_end = (j + 1) * w_step
                                else:
                                    w_end = w
                                input_list.append(x[:, :, h_start:h_end, w_start:w_end])
                        with torch.no_grad():
                            density_sum = 0.0
                            for idx, input in enumerate(input_list):
                                density, seg = self.clf(input)
                                density = density / label_expansion
                                density_sum += density.sum()
                    else:
                        density, seg = self.clf(x)
                        density = density / label_expansion
                        density_sum = density.sum()
                else:
                    density, seg = self.clf(x)
                    density = density / label_expansion
                    density_sum = density.sum()

                density = density.squeeze(0)
                seg_gt_raw = torch.where(y > 0, torch.tensor(1), torch.tensor(0))
                # seg_gt = F.one_hot(seg_gt_raw, num_classes=2).permute(0, 3, 1, 2).float()
                mae = abs(density_sum - y.sum())
                mse = (density_sum - y.sum()) ** 2
                mse_sum += mse.item()
                mae_sum += mae.item()
                # ce_sum += ce_loss(seg, seg_gt).item()
                if y.sum() <= 50:
                    density_eval['low'].append(mae)
                    density_eval_mse['low'].append(mse)
                elif y.sum() <= 500:
                    density_eval['medium'].append(mae)
                    density_eval_mse['medium'].append(mse)
                else:
                    density_eval['high'].append(mae)
                    density_eval_mse['high'].append(mse)

        # Aggregate the results
        if density_eval is not None:
            for key in density_eval.keys():
                if density_eval[key]:
                    density_eval[key] = torch.mean(torch.stack(density_eval[key])).item()

                if density_eval_mse[key]:
                    density_eval_mse[key] = torch.mean(torch.stack(density_eval_mse[key]))
                    density_eval_mse[key] = math.sqrt(density_eval_mse[key].item())

        mae_sum /= len(loader)
        mse_sum /= len(loader)
        mse_sum = math.sqrt(mse_sum)

        return mae_sum, mse_sum, density_eval, density_eval_mse

    def predict(self, data):
        self.clf.eval()
        preds = []
        segs = []
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                density, seg = self.clf(x)
                density = density
                preds.append(density.squeeze(0).squeeze(0).data.cpu().numpy())
                segs.append(torch.sigmoid(seg).squeeze(0).squeeze(0).data.cpu().numpy())
        return preds, segs

    def predict_single(self, data):
        self.clf.eval()
        with torch.no_grad():
            data = data.squeeze(0).float().to(self.device)
            density, _ = self.clf(data)
        return density

    def get_seg_density_predictions(self, data):
        self.clf.eval()
        data = data.float()
        # with torch.no_grad():
            # data = data.to(self.device)
        density, seg = self.clf(data)
        density = density.squeeze(0)
        seg = torch.sigmoid(seg).squeeze(0)

        return density, seg


class Net_Crowd_Counting_SGANet(Net):
    def __init__(self, net, params, device):
        super(Net_Crowd_Counting_SGANet, self).__init__(net, params, device)

    def train(self, labelled_data, test_data):
        n_epoch = self.params['n_epoch']
        self.clf = self.net()
        model_urls = {
            # Inception v3 ported from TensorFlow
            'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
        }
        self.clf.load_state_dict(model_zoo.load_url(model_urls['inception_v3_google']), strict=False)

        if torch.cuda.device_count() > 1:
            self.clf = nn.DataParallel(self.clf)
        self.clf.to(self.device)

        self.clf.train()
        if self.params['optimizer'] == 'Adam':
            optimizer = optim.Adam(self.clf.parameters(), **self.params['optimizer_args'])
            lr_schedule = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        elif self.params['optimizer'] == 'SGD':
            optimizer = optim.SGD(self.clf.parameters(), **self.params['optimizer_args'])
            lr_schedule = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)
        else:
            raise NotImplementedError

        cl_loss = True
        seg_loss = True
        loader_labelled = DataLoader(labelled_data, shuffle=True, **self.params['loader_tr_args'])

        # In crowd counting the batch size is one, so we don't need to divide by it
        criterion1 = nn.MSELoss(reduce=False)  # for density map loss
        criterion2 = nn.BCELoss()  # for segmentation map loss
        best_model_wts = copy.deepcopy(self.clf.state_dict())
        best_mae_val = 1e6
        best_mse_val = 1e6
        best_mae_by_val = 1e6
        best_mae_by_test = 1e6
        best_mse_by_val = 1e6
        best_mse_by_test = 1e6
        min_density_eval = None
        min_density_eval_mse = None
        for epoch in tqdm(range(1, n_epoch + 1), ncols=100):
            self.clf.train()  # Set model to training mode
            running_loss = 0.0
            for index, (inputs, labels, fbs, idx) in enumerate(loader_labelled):
                labels = labels * 100
                labels = skimage.measure.block_reduce(labels.cpu().numpy(), (1, 1, 1, 4, 4), np.sum)
                fbs = skimage.measure.block_reduce(fbs.cpu().numpy(), (1, 1, 1, 4, 4), np.max)
                fbs = np.float32(fbs > 0)
                labels = torch.from_numpy(labels)
                fbs = torch.from_numpy(fbs)
                labels = labels.to(self.device)
                fbs = fbs.to(self.device)
                inputs = inputs.to(self.device)
                inputs = inputs.view(-1, inputs.shape[2], inputs.shape[3], inputs.shape[4])
                labels = labels.view(-1, labels.shape[3], labels.shape[4])
                fbs = fbs.view(-1, fbs.shape[3], fbs.shape[4])
                inputs = inputs.float()
                labels = labels.float()
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(True):
                    output, fbs_out = self.clf(inputs)
                    loss_den = criterion1(output, labels)
                    loss_seg = criterion2(fbs_out, fbs)
                    if cl_loss:
                        th = 0.1 * epoch + 5  # cl2
                    else:
                        th = 1000  # no curriculum loss when th is set a big number
                    weights = th / (F.relu(labels - th) + th)
                    loss_den = loss_den * weights
                    loss_den = loss_den.sum() / weights.sum()
                    if seg_loss:
                        loss = loss_den + 20 * loss_seg
                    else:
                        loss = loss_den

                    loss.backward()
                    optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / len(labelled_data)
            if self.params['optimizer'] == 'Adam':
                lr_schedule.step()
            mae, _, mse, density_eval, density_eval_mse = self.evaluate(test_data)
            print('Epoch: {}/{}, Loss: {:.4f}, MAE: {:.4f}, MSE: {:.4f}, Min MAE: {:.4f}'.format(epoch, n_epoch, epoch_loss,
                                                                                 mae, mse, best_mae_val))
            if mae < best_mae_val:
                best_mae_val = mae
                best_mse_val = mse
                min_density_eval = density_eval
                min_density_eval_mse = density_eval_mse

                # 这里还没写完
        res = {'mae': best_mae_val,
               'mse': best_mse_val,
               'density_eval': min_density_eval,
               'density_eval_mse': min_density_eval_mse,
               'success': True}
        return res

    def evaluate(self, data):
        self.clf.eval()
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        mae_sum = 0
        mse_sum = 0
        ce_sum = 0
        density_eval = {'low': [], 'medium': [], 'high': []}
        density_eval_mse = {'low': [], 'medium': [], 'high': []}
        with torch.no_grad():
            for batch_idx, (x, y, idxs) in enumerate(loader):
                x, y = x.to(self.device).float(), y.to(self.device)
                x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])
                y = y.view(-1, y.shape[3], y.shape[4])
                # y = torch.nn.functional.interpolate(y.unsqueeze(0), size=(y.shape[1] // 8, y.shape[2] // 8),
                #                                     mode='bilinear', align_corners=True).squeeze(0) * 64
                # Try to avoid the out of memory error, using sliding window to predict
                if self.params['name'] == 'JHUCrowd' or self.params['name'] == 'UCF-QNRF':
                    b, c, h, w = x.shape
                    input_list = []
                    if h >= 3584 or w >= 3584:
                        h_stride = int(math.ceil(1.0 * h / 3584))
                        w_stride = int(math.ceil(1.0 * w / 3584))
                        h_step = h // h_stride
                        w_step = w // w_stride
                        for i in range(h_stride):
                            for j in range(w_stride):
                                h_start = i * h_step
                                if i != h_stride - 1:
                                    h_end = (i + 1) * h_step
                                else:
                                    h_end = h
                                w_start = j * w_step
                                if j != w_stride - 1:
                                    w_end = (j + 1) * w_step
                                else:
                                    w_end = w
                                input_list.append(x[:, :, h_start:h_end, w_start:w_end])
                        with torch.no_grad():
                            density_sum = 0.0
                            for idx, input in enumerate(input_list):
                                density, seg = self.clf(input)
                                density = density / 100
                                density_sum += density.sum()
                    else:
                        density, seg  = self.clf(x)
                        density = density / 100
                        density_sum = density.sum()
                else:
                    density, seg = self.clf(x)
                    density = density / 100
                    density_sum = density.sum()

                density = density.squeeze(0)
                density = density / 100
                seg_gt_raw = torch.where(y > 0, torch.tensor(1), torch.tensor(0))
                # seg_gt = F.one_hot(seg_gt_raw, num_classes=2).permute(0, 3, 1, 2).float()
                mae = abs(density_sum - y.sum())
                mse = (density_sum - y.sum()) ** 2
                mse_sum += mse.item()
                mae_sum += mae.item()
                # ce_sum += ce_loss(seg, seg_gt).item()
                if y.sum() <= 50:
                    density_eval['low'].append(mae)
                    density_eval_mse['low'].append(mse)
                elif y.sum() <= 500:
                    density_eval['medium'].append(mae)
                    density_eval_mse['medium'].append(mse)
                else:
                    density_eval['high'].append(mae)
                    density_eval_mse['high'].append(mse)

        mae_sum /= len(loader)
        ce_sum /= len(loader)
        mse_sum /= len(loader)
        mse_sum = math.sqrt(mse_sum)
        if density_eval is not None:
            for key in density_eval.keys():
                if density_eval[key]:
                    density_eval[key] = torch.mean(torch.stack(density_eval[key])).item()

                if density_eval_mse[key]:
                    density_eval_mse[key] = torch.mean(torch.stack(density_eval_mse[key]))
                    density_eval_mse[key] = math.sqrt(density_eval_mse[key].item())

        return mae_sum, ce_sum, mse_sum, density_eval, density_eval_mse

    def get_seg_density_predictions(self, data):
        self.clf.eval()
        data = data.squeeze(0).float()
        density, seg = self.clf(data)
        # TODO
        density_original = density / 100
        seg = torch.log(seg / (1 - seg))
        return density_original, seg

    def predict(self, data):
        self.clf.eval()
        preds = []
        segs = []
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x = x.squeeze(0).float()
                x, y = x.to(self.device), y.to(self.device)
                density, seg = self.clf(x)
                density = density / 100
                preds.append(density.squeeze(0).squeeze(0).data.cpu().numpy())
                segs.append(torch.sigmoid(seg).squeeze(0).squeeze(0).data.cpu().numpy())
        return preds, segs

    def predict_single(self, data):
        self.clf.eval()
        with torch.no_grad():
            if data.dim() > 4:
                data = data.squeeze(0)
            data = data.float().to(self.device)
            density, _ = self.clf(data)
        return density / 100

# Not used in the project!
class Net_Crowd_Counting_Semi(Net):
    def __init__(self, net, params, device):
        super(Net_Crowd_Counting_Semi, self).__init__(net, params, device)
        self.lambda_seg = 0.5
        self.lambda_con = 1
        self.lambda_u = 0.5
        self.T_c = 20
        self.MaxIter = 4

    def train(self, labelled_data, labelled_data_partial, unlabeled_data_partial, test_data, handler):
        n_epoch = self.params['n_epoch']

        ckpt_path = 'csrnet_semi.pth'

        self.clf = self.net()

        if torch.cuda.device_count() > 1:
            self.clf = nn.DataParallel(self.clf)
        self.clf.to(self.device)

        self.clf.train()
        if self.params['optimizer'] == 'Adam':
            optimizer = optim.Adam(self.clf.parameters(), **self.params['optimizer_args'])
            lr_schedule = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)
        elif self.params['optimizer'] == 'SGD':
            optimizer = optim.SGD(self.clf.parameters(), **self.params['optimizer_args'])
            lr_schedule = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)
        else:
            raise NotImplementedError

        loader_labelled = DataLoader(labelled_data, shuffle=True, **self.params['loader_tr_args'])

        loader_semi = DataLoader(handler(labelled_data_partial[0], labelled_data_partial[1], unlabeled_data_partial[0],
                                         unlabeled_data_partial[1], transform=self.params['transform_train'],
                                         resize=True, data_ratio=1),
                                 shuffle=True, **self.params['loader_tr_args'])

        min_mae = sys.maxsize
        # In crowd counting the batch size is one, so we don't need to divide by it
        criterion = nn.MSELoss(reduction='sum')
        conf_criterion = nn.MSELoss(reduction='sum')
        ce_loss = nn.CrossEntropyLoss(reduction='sum')
        best_epoch = 0
        for epoch in tqdm(range(1, n_epoch + 1), ncols=100):
            if self.params['optimizer'] == 'Adam':
                lr_schedule.step()
            # Training labelled data
            for batch_idx, (x, y, idxs) in enumerate(loader_labelled):
                x, y = x.to(self.device), y.to(self.device)
                # Resize y to match the output of the network
                y = torch.nn.functional.interpolate(y.unsqueeze(0), size=(y.shape[1] // 8, y.shape[2] // 8),
                                                    mode='bilinear', align_corners=True).squeeze(0) * 64

                seg_gt_raw = torch.where(y > 0, torch.tensor(1), torch.tensor(0))
                seg_gt = F.one_hot(seg_gt_raw, num_classes=2).permute(0, 3, 1, 2).float()
                optimizer.zero_grad()
                density, seg, conf = self.clf(x)
                out = density.squeeze(1)
                # Calculate density loss
                density_loss = criterion(out, y)
                # Calculate segmentation loss
                seg_loss = ce_loss(seg, seg_gt)
                # Calculate confidence loss
                if epoch > self.T_c:
                    conf_gt = seg.gather(dim=1, index=seg_gt_raw.unsqueeze(1))
                    conf_loss = conf_criterion(conf, conf_gt)
                else:
                    conf_loss = torch.tensor(0)

                # Calculate total loss
                total_loss = density_loss + self.lambda_seg * seg_loss + self.lambda_con * conf_loss

                total_loss.backward()
                optimizer.step()

            # evaluate
            mae = self.evaluate(test_data, epoch)
            if mae < min_mae:
                min_mae = mae
                best_epoch = epoch
                # Update the model
                if mae < 90:
                    self.update_model(self.clf, ckpt_path)
            print('Epoch: {}, MAE: {:.4f}, Min MAE: {:.4f}'.format(epoch, mae, min_mae))

        # For debug purpose
        # ckpt = {'net': self.clf.state_dict()}
        # torch.save(ckpt, 'ckpt_best.pth')
        # self.clf = CSR_Net_Semi()
        # self.clf.load_state_dict(torch.load('ckpt_best.pth')['net'])
        # self.clf.to(self.device)

        # Training unlabelled data
        prev_net = self.clf
        del self.clf
        del self.net
        for i in range(self.MaxIter):
            print('Iteration: {}'.format(i))
            # Loading the best model
            print('Loading the best model...')
            del prev_net, optimizer, lr_schedule, density, seg, conf
            torch.cuda.empty_cache()
            gc.collect()

            with torch.no_grad():
                checkpoint = torch.load(f'ckpt/{ckpt_path}')
                prev_net = CSR_Net_Semi()
                prev_net.load_state_dict(checkpoint['net'])
                prev_net.to(self.device)
                prev_net.eval()

            print('Generating pseudo labels...')
            pseudo_label_density, pseudo_label_idx = self.pseudo_label_generation(prev_net, loader_semi)
            # Delete previous model
            del prev_net
            torch.cuda.empty_cache()
            gc.collect()

            net_new = CSR_Net_Semi()
            net_new.to(device=self.device)
            if self.params['optimizer'] == 'Adam':
                optimizer = optim.Adam(net_new.parameters(), **self.params['optimizer_args'])
                lr_schedule = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)
            elif self.params['optimizer'] == 'SGD':
                optimizer = optim.SGD(net_new.parameters(), **self.params['optimizer_args'])
                lr_schedule = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)
            else:
                raise NotImplementedError
            net_new.train()
            for epoch in tqdm(range(1, n_epoch + 1), ncols=100):
                if self.params['optimizer'] == 'Adam':
                    lr_schedule.step()

                for batch_idx, (x, y, labelled, idxs) in enumerate(loader_semi):
                    x, y = x.to(self.device), y.to(self.device)
                    optimizer.zero_grad()
                    density, seg, conf = net_new(x)
                    # Training labelled data
                    if labelled:
                        seg_gt_raw = torch.where(y > 0, torch.tensor(1), torch.tensor(0))
                        seg_gt = F.one_hot(seg_gt_raw, num_classes=2).permute(0, 3, 1, 2).float()
                        out = density.squeeze(1)
                        # Calculate density loss
                        density_loss = criterion(out, y)

                        # Calculate segmentation loss
                        seg_loss = ce_loss(seg, seg_gt)
                        # Calculate confidence loss
                        if epoch > self.T_c:
                            conf_gt = seg.gather(dim=1, index=seg_gt_raw.unsqueeze(1))
                            conf_loss = conf_criterion(conf, conf_gt)
                        else:
                            conf_loss = torch.tensor(0)

                        # Calculate total loss
                        total_loss = density_loss + self.lambda_seg * seg_loss + self.lambda_con * conf_loss
                    # Training unlabelled data
                    else:
                        # print('Training unlabelled data...')
                        # print(density.sum())
                        out = density.squeeze(1)
                        # Calculate density loss
                        pseudo_idx = pseudo_label_idx.index(idxs)
                        pseudo_label = pseudo_label_density[pseudo_idx]
                        density_loss = criterion(out, pseudo_label)
                        # Calculate total loss
                        total_loss = self.lambda_u * density_loss

                    total_loss.backward()
                    optimizer.step()

                # evaluate
                mae = self.evaluate(test_data, epoch, itera=i, net=net_new)
                if mae < min_mae:
                    min_mae = mae
                    if mae < 90:
                        self.update_model(self.clf, ckpt_path)
                print('Epoch: {}, MAE: {:.4f}, Min MAE: {:.4f}'.format(epoch, mae, min_mae))

            prev_net = net_new
            net_new = None

        self.net = prev_net
        self.clf = self.net

        res = {'mae': min_mae}
        return res

    def update_model(self, model, ckpt_path):
        if not os.path.exists('ckpt'):
            os.mkdir('ckpt')

        ckpt = {'net': model.state_dict()}
        torch.save(ckpt, f'ckpt/{ckpt_path}')

    def pseudo_label_generation(self, net, loader_pseudo):
        pseudo_labels = []
        pseudo_labels_idx = []
        net.eval()
        with torch.no_grad():
            for batch_idx, (x, y, labelled, idxs) in tqdm(enumerate(loader_pseudo)):
                if labelled:
                    continue
                x, y = x.to(self.device), y.to(self.device)
                density, seg, conf = net(x)
                pseudo_label = self.generate_pseudo_label(density, seg, conf).squeeze(1)
                pseudo_labels.append(pseudo_label)
                pseudo_labels_idx.append(idxs)

        return pseudo_labels, pseudo_labels_idx

    def generate_pseudo_label(self, density, seg, conf):
        conf = conf.squeeze(0)
        seg_pred = torch.argmax(seg, dim=1)
        seg_foreground = torch.where(seg_pred > 0.5, torch.tensor(1.0), torch.tensor(0.0))
        seg_background = torch.where(seg_pred > 0.5, torch.tensor(0.0), torch.tensor(1.0))
        conf_foreground = conf * seg_foreground
        conf_background = conf * seg_background

        # Normalize confidence
        num_pixel = conf_background.numel()

        conf_foreground /= (torch.sum(conf_foreground) + 1e-8)
        conf_background /= (torch.sum(conf_background) + 1e-8)
        conf_map = conf_foreground + conf_background
        conf_map *= num_pixel

        # Generate pseudo label
        pseudo_label = density * conf_map
        return pseudo_label

    def evaluate(self, data, epoch, itera=0, net=None):
        if net is None:
            self.clf.eval()
        else:
            net.eval()
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        mae_sum = 0
        import csv
        import os
        header = ['confidence', 'density', 'pseudo_label', 'ground_truth']
        rows = []
        with torch.no_grad():
            for batch_idx, (x, y, idxs) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)
                if net is None:
                    density, seg, confidence = self.clf(x)
                else:
                    density, seg, confidence = net(x)
                mae = abs(density.sum() - y.sum())
                mae_sum += mae.item()
                if epoch % 50 == 0:
                    pl = self.generate_pseudo_label(density, seg, confidence)
                    rows.append([confidence.sum().item(), density.sum().item(), pl.sum().item(), y.sum().item()])
        if epoch % 50 == 0:
            if not os.path.exists('res_semi'):
                os.makedirs('res_semi')
            with open(f'res_semi/1e6results_iter-{itera}_epoch-{epoch}.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(rows)
        mae_sum /= len(loader)
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


class Net_Crowd_Counting_IRAST(Net):
    """
    Follow the implementation in https://github.com/leolyj/IRAST
    """

    def __init__(self, net, params, device):
        super(Net_Crowd_Counting_IRAST, self).__init__(net, params, device)

    def train(self, data, unlabeled_data, test_data, handler=None):
        n_epoch = self.params['n_epoch']

        self.clf = self.net()

        if torch.cuda.device_count() > 1:
            self.clf = nn.DataParallel(self.clf)
        self.clf.to(self.device)

        self.clf.train()
        if self.params['optimizer'] == 'Adam':
            optimizer = optim.Adam(self.clf.parameters(), **self.params['optimizer_args'])
        elif self.params['optimizer'] == 'SGD':
            optimizer = optim.SGD(self.clf.parameters(), **self.params['optimizer_args'])
            lr_schedule = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        else:
            raise NotImplementedError

        loader = DataLoader(handler(data[0], data[1], unlabeled_data[0], unlabeled_data[1],
                                    transform=self.params['transform_train']), shuffle=True,
                            **self.params['loader_tr_args'])

        min_mae = sys.maxsize
        # In crowd counting the batch size is one, so we don't need to divide by it
        criterion = nn.MSELoss(reduction='sum')
        cls_criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=10)
        for epoch in tqdm(range(1, n_epoch + 1), ncols=100):
            if self.params['optimizer'] == 'SGD':
                lr_schedule.step()
            for batch_idx, (x, y, labelled, idxs) in enumerate(loader):
                x, y = x.to(self.device).float(), y.to(self.device)
                optimizer.zero_grad()
                out, l2, l3, l4 = self.clf(x)
                y = F.relu(y)

                if labelled:
                    target_mask_2 = self.densitymap_to_densitymask(y, threshold=0.0)
                    target_mask_3 = self.densitymap_to_densitymask(y, threshold=0.048)
                    target_mask_4 = self.densitymap_to_densitymask(y, threshold=0.258)

                    loss_cls_label2 = cls_criterion(l2, target_mask_2)
                    loss_cls_label3 = cls_criterion(l3, target_mask_3)
                    loss_cls_label4 = cls_criterion(l4, target_mask_4)

                    out = out.squeeze(1)
                    loss_density = criterion(out, y)

                    loss = 1 * loss_density + 0.01 * (
                            loss_cls_label2.sum() + loss_cls_label3.sum() + loss_cls_label4.sum()) / 3.0
                else:
                    pro2u, pro3u, pro4u = F.softmax(l2, dim=1), F.softmax(l3, dim=1), F.softmax(l4, dim=1)

                    loss_cls_unlabel2 = self.unlabel_CE_loss2v1(logits2=l2, prob3=pro3u, prob4=pro4u, th=0.9,
                                                                max_update_pixel=0,
                                                                criterion_cls=cls_criterion)
                    loss_cls_unlabel3 = self.unlabel_CE_loss3v1(prob2=pro2u, logits3=l3, prob4=pro4u, th=0.9,
                                                                max_update_pixel=0,
                                                                criterion_cls=cls_criterion)
                    loss_cls_unlabel4 = self.unlabel_CE_loss4v1(prob2=pro2u, prob3=pro3u, logits4=l4, th=0.9,
                                                                max_update_pixel=0,
                                                                criterion_cls=cls_criterion)
                    loss = 0.01 * (loss_cls_unlabel2.sum() + loss_cls_unlabel3.sum() + loss_cls_unlabel4.sum()) / 3.0

                loss.backward()
                optimizer.step()

            # evaluate
            mae = self.evaluate(test_data)
            if mae < min_mae:
                min_mae = mae
            print('Epoch: {}, MAE: {:.4f}, Min MAE: {:.4f}'.format(epoch, mae, min_mae))

        res = {'mae': min_mae}
        return res

    def unlabel_CE_loss2v1(self, logits2, prob3, prob4, max_update_pixel, th, criterion_cls):
        prob2 = torch.nn.functional.softmax(logits2, dim=1)
        prob_max = torch.max(prob2, dim=1)[0]
        target_temp = torch.argmax(prob2, dim=1)

        # choose the valid pixels
        mask = ((prob2[:, 1, :, :] > th) |
                ((prob2[:, 0, :, :] > th) & (prob3[:, 0, :, :] > th) & (prob4[:, 0, :, :] > th))).float()

        target = ((mask * target_temp.float()) + (10 * (1 - mask))).long()

        loss_ce_ori = criterion_cls(logits2, target)
        return loss_ce_ori

    def unlabel_CE_loss3v1(self, logits3, prob2, prob4, max_update_pixel, th, criterion_cls):
        prob3 = torch.nn.functional.softmax(logits3, dim=1)
        prob_max = torch.max(prob3, dim=1)[0]
        target_temp = torch.argmax(prob3, dim=1)

        # choose the valid pixels
        mask = (((prob2[:, 1, :, :] > th) & (prob3[:, 1, :, :] > th)) |
                ((prob3[:, 0, :, :] > th) & (prob4[:, 0, :, :] > th))).float()

        target = ((mask * target_temp.float()) + (10 * (1 - mask))).long()
        loss_ce_ori = criterion_cls(logits3, target)
        return loss_ce_ori

    def unlabel_CE_loss4v1(self, logits4, prob2, prob3, max_update_pixel, th, criterion_cls):
        prob4 = torch.nn.functional.softmax(logits4, dim=1)
        prob_max = torch.max(prob4, dim=1)[0]
        target_temp = torch.argmax(prob4, dim=1)

        # choose the valid pixels
        mask = (((prob2[:, 1, :, :] > th) & (prob3[:, 1, :, :] > th) & (prob4[:, 1, :, :] > th))
                | ((prob4[:, 0, :, :] > th))).float()

        target = ((mask * target_temp.float()) + (10 * (1 - mask))).long()
        loss_ce_ori = criterion_cls(logits4, target)
        return loss_ce_ori

    def densitymap_to_densitymask(self, density_map, threshold):
        density_mask = (density_map > threshold).long()
        return density_mask

    def evaluate(self, data):
        self.clf.eval()
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        mae_sum = 0

        with torch.no_grad():
            for batch_idx, (x, y, idxs) in enumerate(loader):
                x, y = x.to(self.device).float(), y.to(self.device)
                out, _, _, _ = self.clf(x)
                mae = abs(out.sum() - y.sum())
                mae_sum += mae.item()
        mae_sum /= len(loader)
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


class CSR_Net(nn.Module):
    def __init__(self, load_weights=False):
        super(CSR_Net, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
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
        x = self.output_layer(x)
        x = F.interpolate(x, scale_factor=8)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class CSR_Net_PSSW(nn.Module):
    def __init__(self, load_weights=False):
        super(CSR_Net_PSSW, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self.grl = GradientReversalLayer()
        self.mix_layer = MixupLayer()
        self.conv1 = nn.Conv2d(512, 1, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool2d(1)

        if not load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            items = list(self.frontend.state_dict().items())
            _items = list(mod.state_dict().items())
            for i in range(len(self.frontend.state_dict().items())):
                items[i][1].data[:] = _items[i][1].data[:]

    def forward(self, x, y=None):
        fx = self.frontend(x)

        if y is not None:
            # First prevent unlabelled data to density regression
            idx = torch.where(x == 1)[0]
            if len(idx) > 0:
                filtered_fx = fx[idx]
                filtered_fx = self.backend(filtered_fx)
                filtered_fx = self.output_layer(filtered_fx)
                filtered_fx = F.interpolate(filtered_fx, scale_factor=8)
            else:
                filtered_fx = None

            xx = self.grl(fx)
            xx, y = self.mix_layer(xx, y)
            xx = self.conv1(xx)
            xx = self.gap(xx)
            return filtered_fx, xx, y
        else:
            fx = self.backend(fx)
            fx = self.output_layer(fx)
            fx = F.interpolate(fx, scale_factor=8)
            return fx

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


class GradientReversalLayer(torch.nn.Module):
    def __init__(self):
        super(GradientReversalLayer, self).__init__()

    def forward(self, x):
        return GradientReversalFunction.apply(x)


class MixupLayer(torch.nn.Module):
    def __init__(self, alpha=0.5):
        super(MixupLayer, self).__init__()
        self.alpha = alpha

    def forward(self, x, y):
        lam = np.random.beta(self.alpha, self.alpha)
        lam = max(lam, 1 - lam)
        x = lam * x[0] + (1 - lam) * x[1]
        y = lam * y[0] + (1 - lam) * y[1]
        return x, y


class CSR_Net_Semi(nn.Module):
    def __init__(self, load_weights=False):
        super(CSR_Net_Semi, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256]
        self.frontend = make_layers(self.frontend_feat)
        self.shared_feat = make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.backend = make_layers([128, 64], in_channels=256, dilation=True)
        self.seg_cof_extrator = nn.Conv2d(256, 128, kernel_size=3, padding=2, dilation=2)
        self.segmentation_layer = SegmentationLayer()
        self.confidence_layer = ConfidenceLayer()
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        if not load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            items = list(self.frontend.state_dict().items())
            _items = list(mod.state_dict().items())
            for i in range(len(self.frontend.state_dict().items())):
                items[i][1].data[:] = _items[i][1].data[:]

    def forward(self, x):
        x = self.frontend(x)
        feat = self.shared_feat(x)

        # Density network
        density = self.backend(feat)
        density = self.output_layer(density)
        density = F.relu(density)

        # Segmentation network
        seg_cof_feat = self.seg_cof_extrator(feat)
        seg_map = self.segmentation_layer(seg_cof_feat)

        # Confidence network
        confidence = self.confidence_layer(seg_cof_feat)

        # density = F.interpolate(density, scale_factor=8)
        # seg_map = F.interpolate(seg_map, scale_factor=8)
        # confidence = F.interpolate(confidence, scale_factor=8)
        return density, seg_map, confidence

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class CSR_Net_Project(nn.Module):
    def __init__(self, load_weights=False):
        super(CSR_Net_Project, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256]
        self.frontend = make_layers(self.frontend_feat)
        self.shared_feat = make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.backend = make_layers([128, 64], in_channels=256, dilation=True)
        self.seg_cof_extrator = nn.Conv2d(256, 128, kernel_size=3, padding=2, dilation=2)
        self.segmentation_layer = SegmentationLayer()
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        if not load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            items = list(self.frontend.state_dict().items())
            _items = list(mod.state_dict().items())
            for i in range(len(self.frontend.state_dict().items())):
                items[i][1].data[:] = _items[i][1].data[:]

    def forward(self, x):
        x = self.frontend(x)
        feat = self.shared_feat(x)

        # Density network
        density_feat = self.backend(feat)
        density = self.output_layer(density_feat)
        density = F.relu(density)

        # Segmentation network
        seg_cof_feat = self.seg_cof_extrator(feat)
        seg_map = self.segmentation_layer(seg_cof_feat)

        density = F.interpolate(density, scale_factor=8)
        seg_map = F.interpolate(seg_map, scale_factor=8)
        return density, seg_map

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class CSR_Net_Project1(nn.Module):
    def __init__(self, load_weights=False):
        super(CSR_Net_Project1, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256]
        self.frontend = make_layers(self.frontend_feat)
        self.shared_feat = make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.backend = make_layers([128, 64], in_channels=256, dilation=True)
        self.backend_uncertainty = make_layers([128, 64], in_channels=256, dilation=True)
        self.seg_cof_extrator = nn.Conv2d(256, 128, kernel_size=3, padding=2, dilation=2)
        self.segmentation_layer = SegmentationLayer()
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        self.uncertainty_layer = nn.Conv2d(64, 1, kernel_size=1)
        if not load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            items = list(self.frontend.state_dict().items())
            _items = list(mod.state_dict().items())
            for i in range(len(self.frontend.state_dict().items())):
                items[i][1].data[:] = _items[i][1].data[:]

    def forward(self, x):
        x = self.frontend(x)
        feat = self.shared_feat(x)

        # Density network
        density_feat = self.backend(feat)
        density = self.output_layer(density_feat)
        density = F.relu(density)

        # Uncertainty network
        uncertainty_feat = self.backend_uncertainty(feat)
        uncertainty = self.uncertainty_layer(uncertainty_feat)
        uncertainty = F.relu(uncertainty)
        # uncertainty = torch.clamp(uncertainty, 1e-8, 100)

        # Segmentation network
        seg_cof_feat = self.seg_cof_extrator(feat)
        seg_map = self.segmentation_layer(seg_cof_feat)

        density = F.interpolate(density, scale_factor=8)
        seg_map = F.interpolate(seg_map, scale_factor=8)
        uncertainty = F.interpolate(uncertainty, scale_factor=8)
        return density, uncertainty, seg_map

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class SegmentationLayer(nn.Module):
    def __init__(self):
        super(SegmentationLayer, self).__init__()
        self.conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=2, dilation=2)
        self.conv2 = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        return x


class ConfidenceLayer(nn.Module):
    def __init__(self):
        super(ConfidenceLayer, self).__init__()
        self.conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 400, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(400, 120, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(120, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        return F.sigmoid(x)


class CSR_Net_IRAST(nn.Module):
    """
    Follow the implementation in https://github.com/leolyj/IRAST
    """

    def __init__(self, load_weights=False):
        super(CSR_Net_IRAST, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat_0 = [512, 512, 512, 256]

        self.backend_0 = make_layers(self.backend_feat_0, in_channels=512, batch_norm=False, dilation=True)

        self.frontend = make_layers(self.frontend_feat)

        # density regressor
        self.backend_feat_1 = [128, 64]
        self.backend_1 = make_layers(self.backend_feat_1, in_channels=256, batch_norm=False, dilation=True)

        self.output_layer_1 = nn.Conv2d(64, 1, kernel_size=1)

        # 3 surrogate tasks
        self.backend_feat_2 = [128, 64]
        self.backend_feat_3 = [128, 64]
        self.backend_feat_4 = [128, 64]

        self.backend_2 = make_layers(self.backend_feat_2, in_channels=256, batch_norm=False, dilation=True)
        self.backend_3 = make_layers(self.backend_feat_3, in_channels=256, batch_norm=False, dilation=True)
        self.backend_4 = make_layers(self.backend_feat_4, in_channels=256, batch_norm=False, dilation=True)

        self.output_layer_2 = nn.Conv2d(64, 2, kernel_size=1)
        self.output_layer_3 = nn.Conv2d(64, 2, kernel_size=1)
        self.output_layer_4 = nn.Conv2d(64, 2, kernel_size=1)
        if not load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            items = list(self.frontend.state_dict().items())
            _items = list(mod.state_dict().items())
            for i in range(len(self.frontend.state_dict().items())):
                items[i][1].data[:] = _items[i][1].data[:]

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend_0(x)

        u = self.backend_1(x)
        u = self.output_layer_1(u)

        u = F.relu(u)

        logits2 = self.backend_2(x)
        logits2 = self.output_layer_2(logits2)

        logits3 = self.backend_3(x)
        logits3 = self.output_layer_3(logits3)

        logits4 = self.backend_4(x)
        logits4 = self.output_layer_4(logits4)

        # u is the predicted density map,logits is the prediction of surrogate tasks
        return u, logits2, logits3, logits4

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


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
    def __init__(self, block=Bottleneck, num_blocks=None, load_weights=False):
        super(DUBNet, self).__init__()
        # Resnet 50
        if num_blocks is None:
            num_blocks = [3, 4, 6]
        self.in_planes = 64

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
        self.outputs = [nn.Conv2d(64, 1, kernel_size=1) for _ in range(8)]
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
            # Randomly pick head in self.outputs
            idx = random.randint(0, len(self.outputs) - 1)
            density = self.outputs[idx](x)
        else:
            # Average over all heads
            density = torch.stack([head(x) for head in self.outputs], dim=0).mean(dim=0)

        logvar = self.logvar(x)
        density = F.interpolate(density, scale_factor=16, mode='bilinear', align_corners=False)
        logvar = F.interpolate(logvar, scale_factor=16, mode='bilinear', align_corners=False)
        return density, logvar


class SGANet(nn.Module):
    """
    Crowd Counting via Segmentation Guided Attention Networks and Curriculum Loss

    code directly taken from https://github.com/hellowangqian/sganet-crowd-counting with little modifications.
    """
    def __init__(self, num_classes=1000, aux_logits=False, transform_input=False):
        super(SGANet, self).__init__()
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


class CANNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CANNet, self).__init__()
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat, in_channels=1024, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self.conv1_1 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv1_2 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv2_1 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv2_2 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv3_1 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv3_2 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv6_1 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv6_2 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        if not load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            #            print("VGG",list(mod.state_dict().items())[0][1])#要的VGG值
            fsd = collections.OrderedDict()
            for i in range(len(self.frontend.state_dict().items())):  # 10个卷积*（weight，bias）=20个参数
                temp_key = list(self.frontend.state_dict().items())[i][0]
                fsd[temp_key] = list(mod.state_dict().items())[i][1]
            self.frontend.load_state_dict(fsd)

    #            print("Mine",list(self.frontend.state_dict().items())[0][1])#将VGG值赋予自己网络后输出验证
    #            self.frontend.state_dict().items()[i][1].data[:] = mod.state_dict().items()[i][1].data[:]#python2.7版本
    def forward(self, x):
        fv = self.frontend(x)
        # S=1
        ave1 = nn.functional.adaptive_avg_pool2d(fv, (1, 1))
        ave1 = self.conv1_1(ave1)
        #        ave1=nn.functional.relu(ave1)
        s1 = nn.functional.interpolate(ave1, size=(fv.shape[2], fv.shape[3]), mode='bilinear')
        c1 = s1 - fv
        w1 = self.conv1_2(c1)
        w1 = nn.functional.sigmoid(w1)
        # S=2
        ave2 = nn.functional.adaptive_avg_pool2d(fv, (2, 2))
        ave2 = self.conv2_1(ave2)
        #        ave2=nn.functional.relu(ave2)
        s2 = nn.functional.interpolate(ave2, size=(fv.shape[2], fv.shape[3]), mode='bilinear')
        c2 = s2 - fv
        w2 = self.conv2_2(c2)
        w2 = nn.functional.sigmoid(w2)
        # S=3
        ave3 = nn.functional.adaptive_avg_pool2d(fv, (3, 3))
        ave3 = self.conv3_1(ave3)
        #        ave3=nn.functional.relu(ave3)
        s3 = nn.functional.interpolate(ave3, size=(fv.shape[2], fv.shape[3]), mode='bilinear')
        c3 = s3 - fv
        w3 = self.conv3_2(c3)
        w3 = nn.functional.sigmoid(w3)
        # S=6
        #        print('fv',fv.mean())
        ave6 = nn.functional.adaptive_avg_pool2d(fv, (6, 6))
        #        print('ave6',ave6.mean())
        ave6 = self.conv6_1(ave6)
        #        print(ave6.mean())
        #        ave6=nn.functional.relu(ave6)
        s6 = nn.functional.interpolate(ave6, size=(fv.shape[2], fv.shape[3]), mode='bilinear')
        #        print('s6',s6.mean(),'s1',s1.mean(),'s2',s2.mean(),'s3',s3.mean())
        c6 = s6 - fv
        #        print('c6',c6.mean())
        w6 = self.conv6_2(c6)
        w6 = nn.functional.sigmoid(w6)
        #        print('w6',w6.mean())

        fi = (w1 * s1 + w2 * s2 + w3 * s3 + w6 * s6) / (w1 + w2 + w3 + w6 + 0.000000000001)
        #        print('fi',fi.mean())
        #        fi=fv
        x = torch.cat((fv, fi), 1)

        x = self.backend(x)
        x = self.output_layer(x)
        x = nn.functional.interpolate(x, scale_factor=8, mode='bilinear')
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def init_distributed_mode():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ.get('WORLD_SIZE', torch.cuda.device_count()))  # Assuming all GPUs are used if WORLD_SIZE is not set
        gpu = rank % torch.cuda.device_count()
    else:
        rank = 0
        world_size = 2
        gpu = 0
        # return distributed, None

    distributed = True
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'
    torch.cuda.set_device(gpu)
    dist_url = "env://"  # Using environment variables to initialize distributed environment
    dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(rank, dist_url), flush=True)
    dist.init_process_group(backend=dist_backend, init_method=dist_url, world_size=world_size, rank=rank)
    dist.barrier()
    setup_for_distributed(rank == 0)
    return distributed, rank

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print