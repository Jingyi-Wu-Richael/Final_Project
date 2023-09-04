import numpy as np
import torch
from .strategy import Strategy
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

import random
'''
This implementation is with reference of https://github.com/Mephisto405/Learning-Loss-for-Active-Learning.
Please cite the original paper if you use this method.
@inproceedings{yoo2019learning,
  title={Learning loss for active learning},
  author={Yoo, Donggeun and Kweon, In So},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={93--102},
  year={2019}
}
'''


class LossPredictionLoss(Strategy):
    def __init__(self, dataset, net, args_input, args_task):
        super(LossPredictionLoss, self).__init__(dataset, net, args_input, args_task)
        self.subset = 10000
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data_subset(self.subset)
        uncertainties = self.unc_lpl(unlabeled_data)
        return unlabeled_idxs[uncertainties.sort(descending=True)[1][:n]]

    def unc_lpl(self, data):
        # Get subset
        loader = DataLoader(data, shuffle=False, **self.args_task['loader_te_args'])

        self.net.clf.eval()
        self.net.net_lpl.eval()
        uncertainty = torch.tensor([]).to(self.device)
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, feature = self.net.clf(x)
                pred_loss = self.net.net_lpl(feature)
                pred_loss = pred_loss.view(pred_loss.size(0))
                uncertainty = torch.cat((uncertainty, pred_loss), 0)

        uncertainty = uncertainty.cpu()
        return uncertainty
