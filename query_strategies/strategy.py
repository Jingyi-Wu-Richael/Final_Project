import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.optim as optim
from torch.utils.data import DataLoader
from handlers import ShanghaitechHandler_Semi


class Strategy:
    def __init__(self, dataset, net, args_input, args_task):
        self.dataset = dataset
        self.net = net
        self.args_input = args_input
        self.args_task = args_task

    def query(self, n):
        pass

    def get_labeled_count(self):
        labeled_idxs, labeled_data = self.dataset.get_labeled_data()
        return len(labeled_idxs)

    def get_model(self):
        return self.net.get_model()

    def update(self, pos_idxs, neg_idxs=None):
        self.dataset.labeled_idxs[pos_idxs] = True
        if neg_idxs:
            self.dataset.labeled_idxs[neg_idxs] = False

    def train(self, data=None, model_name=None):
        # Returns the best result of the model on the validation set
        if model_name is None:
            if data is None:
                labeled_idxs, labeled_data = self.dataset.get_labeled_data()
                test_data = self.dataset.get_test_data()

                res = self.net.train(labeled_data, test_data)
                # Solve the error when some initialization weights can not be trained on shanghaitechB dataset
                while self.args_task['name'] == 'shanghaitechB' and not res['success']:
                    res = self.net.train(labeled_data, test_data)
                return res
            else:
                return self.net.train(data)
        else:
            # Place for using unlabeled data for training
            if model_name == 'WAAL':
                labeled_idxs, labeled_data = self.dataset.get_labeled_data()
                X_labeled, Y_labeled = self.dataset.get_partial_labeled_data()
                X_unlabeled, Y_unlabeled = self.dataset.get_partial_unlabeled_data()
                return self.net.train(labeled_data, X_labeled, Y_labeled, X_unlabeled, Y_unlabeled)
            # elif model_name == 'CrowdProject' or model_name == 'HeadCount':
            #     labeled_idx, labeled_data = self.dataset.get_labeled_data()
            #     test_data = self.dataset.get_test_data()
            #     data = self.dataset.get_train_data()
            #     print(labeled_idx)
            #
            #     res = self.net.train(labeled_data, test_data, all_data=data, labeled_idx=labeled_idx)
            #     # Solve the error when some initialization weights can not be trained on shanghaitechB dataset
            #     while self.args_task['name'] == 'shanghaitechB' and not res['success']:
            #         res = self.net.train(labeled_data, test_data, all_data=data, labeled_idx=labeled_idx)
            #     return res
            else:
                raise NotImplementedError

    def predict(self, data):
        preds = self.net.predict(data)
        return preds

    def predict_prob(self, data):
        probs = self.net.predict_prob(data)
        return probs

    def predict_prob_dropout(self, data, n_drop=10):
        probs = self.net.predict_prob_dropout(data, n_drop=n_drop)
        return probs

    def predict_prob_dropout_split(self, data, n_drop=10):
        probs = self.net.predict_prob_dropout_split(data, n_drop=n_drop)
        return probs

    def get_embeddings(self, data):
        embeddings = self.net.get_embeddings(data)
        return embeddings

    def get_grad_embeddings(self, data):
        embeddings = self.net.get_grad_embeddings(data)
        return embeddings


class ShanghaitechHandler_PSSW(Dataset):
    def __init__(self, labeled_dataset, unlabeled_dataset):
        self.labeled_dataset = labeled_dataset
        self.unlabeled_dataset = unlabeled_dataset

    def __getitem__(self, index):
        if index < len(self.labeled_dataset):
            data, label, _ = self.labeled_dataset[index]
            return data, label, True  # True indicates that this sample is labeled
        else:
            data, label, _ = self.unlabeled_dataset[index - len(self.labeled_dataset)]
            return data, label, False  # False indicates that this sample is not labeled

    def __len__(self):
        return len(self.labeled_dataset) + len(self.unlabeled_dataset)
