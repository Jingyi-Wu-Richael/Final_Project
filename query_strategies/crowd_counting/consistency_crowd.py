from ..strategy import Strategy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import random
import torchvision.transforms.functional as F


class ConsistencyCrowd(Strategy):
    def __init__(self, dataset, net, args_input, args_task):
        super(ConsistencyCrowd, self).__init__(dataset, net, args_input, args_task)
        self.args_input = args_input
        self.args_task = args_task
        self.dataset = dataset
        self.net = net
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        consistency = self.get_consistency(unlabeled_data)
        return unlabeled_idxs[consistency.sort(descending=True)[1][:n]]

    def get_consistency(self, unlabeled_data, reduction='mean'):
        consistency_unlabeled = torch.zeros(len(unlabeled_data))

        # Generate the predictions of the unlabeled data before augmentation
        ref_predictions, _ = self.net.predict(unlabeled_data)
        # Expand one dimension for channel
        ref_predictions = [torch.from_numpy(ref) for ref in ref_predictions]

        # Batch size is default to be 1
        loader_data = DataLoader(unlabeled_data, shuffle=False, **self.args_task['loader_te_args'])
        for batch_idx, (inputs, targets, idx) in enumerate(loader_data):
            augs = []
            augs_predictions = []

            inputs = inputs.to(self.device)

            # For SGANet, should be removed in the future to make code cleaner
            if inputs.dim() >= 4:
                inputs = inputs.squeeze(0)
            # Rotation
            for i in range(3):
                degree = random.randint(30, 330) # 30, 330
                augs.append(F.rotate(inputs, degree))
                augs_predictions.append(F.rotate(ref_predictions[batch_idx].unsqueeze(0), degree).squeeze(0))
            # Horizontal flip
            if np.random.random() > 0.5:
                augs.append(F.hflip(inputs))
                augs_predictions.append(F.hflip(ref_predictions[batch_idx]))
            # Multi-scale cropping
            for i in range(3):
                scale = random.uniform(0.5, 1.0)
                size = int(round(scale * inputs.shape[-2])), int(round(scale * inputs.shape[-1]))
                augs.append(F.crop(inputs, i, i, size[0], size[1]))
                augs_predictions.append(F.crop(ref_predictions[batch_idx], i, i, size[0], size[1]))
            # Calculate the consistency
            consistency = 0

            for i in range(len(augs)):
                pred = self.net.predict_single(augs[i])
                consistency_loss = abs(pred.sum() - augs_predictions[i].sum())
                if reduction == 'mean':
                    consistency += consistency_loss
                elif reduction == 'max':
                    consistency = max(consistency, consistency_loss)
                else:
                    raise NotImplementedError

            if reduction == 'mean':
                consistency /= len(augs)

            consistency_unlabeled[batch_idx] = consistency

        return consistency_unlabeled
