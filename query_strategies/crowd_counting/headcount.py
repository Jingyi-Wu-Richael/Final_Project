from ..strategy import Strategy
import torch
from torch.utils.data import DataLoader


class HeadCount(Strategy):
    """
    HeadCount query strategy for crowd counting, which selects the images with the highest head count.
    """
    def __init__(self, dataset, net, args_input, args_task):
        super(HeadCount, self).__init__(dataset, net, args_input, args_task)
        self.args_input = args_input
        self.args_task = args_task
        self.dataset = dataset
        self.net = net

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        image_head_count = self.get_head_count(unlabeled_data)
        return unlabeled_idxs[image_head_count.sort(descending=True)[1][:n]]

    def get_head_count(self, unlabeled_data):
        image_head_count = torch.zeros(len(unlabeled_data))
        loader_data = DataLoader(unlabeled_data, shuffle=False, **self.args_task['loader_te_args'])

        for batch_idx, (inputs, targets, idx) in enumerate(loader_data):
            prediction = self.net.predict_single(inputs)
            image_head_count[batch_idx] = prediction.sum()
            torch.cuda.empty_cache()

        return image_head_count
