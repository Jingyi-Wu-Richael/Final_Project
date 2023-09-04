from ..strategy import Strategy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np


import gc
import random


class ActiveSegCrowd(Strategy):
    """
    HeadCount query strategy for crowd counting, which selects the images with the highest head count.
    """

    def __init__(self, dataset, net, args_input, args_task):
        super(ActiveSegCrowd, self).__init__(dataset, net, args_input, args_task)
        self.args_input = args_input
        self.args_task = args_task
        self.dataset = dataset
        self.net = net
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.subset_ratio = 6
        self.L_A = 4

    def query(self, n):
        unlabeled_idx, unlabeled_data = self.dataset.get_unlabeled_data()
        # scores = self.get_uncertainty_score(unlabeled_data)
        # return unlabeled_idx[np.argsort(scores)[::-1][:n]]
        return unlabeled_idx[self.get_data(unlabeled_data, n)]

    def get_data(self, unlabeled_data, n, normalize=False):
        loader = DataLoader(unlabeled_data, shuffle=False, **self.args_task['loader_te_args'])

        sim_ge_list = []
        seg_diff_list = []
        density = []
        with torch.no_grad():
            for batch_idx, (x, y, idxs) in enumerate(loader):
                x = x.to(self.device)

                # 好几次取平均
                preds = []
                for _ in range(3):
                    fliped = random.random() > 0.5
                    if fliped:
                        x_h = torch.flip(x, [3])
                    else:
                        x_h = x
                    x_h = x_h + torch.randn_like(x_h) * 0.1

                    y_pred_h, seg_h = self.net.get_seg_density_predictions(x_h)
                    if fliped:
                        y_pred_h = torch.flip(y_pred_h, [2])
                    y_pred_h = y_pred_h.squeeze(0)
                    preds.append(y_pred_h)

                y_pred_h = torch.mean(torch.stack(preds), dim=0)
                y_pred, seg = self.net.get_seg_density_predictions(x)
                y_pred = y_pred.squeeze(0).squeeze(0)
                seg = seg.squeeze(0).squeeze(0)
                # Calculate the similarity between the prediction and the augmented prediction
                pt_sum1 = self.get_patch_sum(y_pred, self.L_A)
                pt_sum2 = self.get_patch_sum(y_pred_h, self.L_A)

                # Simularity
                sim_ge_list.append(torch.sum(torch.abs(pt_sum1 - pt_sum2)).item())

                # seg diff
                # seg = torch.sigmoid(seg)
                y_pred_seg = y_pred * seg
                seg_diff = abs(y_pred_seg - y_pred).sum().item()
                seg_diff_list.append(seg_diff)

                # For memory efficiency
                y_pred = y_pred.detach().cpu()

                density.append(y_pred)

        # nomarlize
        sim_ge_list = np.array(sim_ge_list)
        sim_ge_list = (sim_ge_list - sim_ge_list.min()) / (sim_ge_list.max() - sim_ge_list.min())

        seg_diff_list = np.array(seg_diff_list)
        seg_diff_list = (seg_diff_list - seg_diff_list.min()) / (seg_diff_list.max() - seg_diff_list.min())

        scores = sim_ge_list * seg_diff_list
        candidate_pools = np.argsort(scores)[::-1][:n * self.subset_ratio]

        # Select data from candidate pools according to diversity measure
        unlabeled_data_patch_sum = []

        for i in range(self.L_A + 1):
            patch_temp = []
            for c in candidate_pools:
                patch_temp.append(self.get_patch_sum(density[c], i))
            patch_temp = torch.stack(patch_temp)
            unlabeled_data_patch_sum.append(patch_temp)

        # for memory efficiency
        del density
        gc.collect()
        torch.cuda.empty_cache()

        # Get dissimilarity matrix
        dissimilarity_matrix = self.calculate_all_dissimilarities(unlabeled_data_patch_sum)
        # Apply k-means
        assign_dis_records, centroid = self.kmeans(dissimilarity_matrix, n)

        # Convert the index of centroid to the original index of unlabeled data
        centroid = candidate_pools[centroid]

        return np.array(centroid)

    # Uncertainty only (For ablation study)
    def get_data_8(self, unlabeled_data, n, normalize=False):
        loader = DataLoader(unlabeled_data, shuffle=False, **self.args_task['loader_te_args'])

        sim_ge_list = []
        seg_diff_list = []
        density = []
        with torch.no_grad():
            for batch_idx, (x, y, idxs) in enumerate(loader):
                x = x.to(self.device)

                # 好几次取平均
                preds = []
                for _ in range(3):
                    fliped = random.random() > 0.5
                    if fliped:
                        x_h = torch.flip(x, [3])
                    else:
                        x_h = x
                    x_h = x_h + torch.randn_like(x_h) * 0.1

                    y_pred_h, seg_h = self.net.get_seg_density_predictions(x_h)
                    if fliped:
                        y_pred_h = torch.flip(y_pred_h, [2])
                    y_pred_h = y_pred_h.squeeze(0)
                    preds.append(y_pred_h)

                y_pred_h = torch.mean(torch.stack(preds), dim=0)
                y_pred, seg = self.net.get_seg_density_predictions(x)
                y_pred = y_pred.squeeze(0).squeeze(0)
                seg = seg.squeeze(0).squeeze(0)

                # Calculate the similarity between the prediction and the augmented prediction
                pt_sum1 = self.get_patch_sum(y_pred, self.L_A)
                pt_sum2 = self.get_patch_sum(y_pred_h, self.L_A)

                sim_ge_list.append(torch.sum(torch.abs(pt_sum1 - pt_sum2)).item())

                # seg diff
                seg = torch.sigmoid(seg)
                y_pred_seg = y_pred * torch.sigmoid(seg)
                seg_diff = abs(y_pred_seg - y_pred).sum().item()
                seg_diff_list.append(seg_diff)
                density.append(y_pred)

        # nomarlize
        sim_ge_list = np.array(sim_ge_list)
        sim_ge_list = (sim_ge_list - sim_ge_list.min()) / (sim_ge_list.max() - sim_ge_list.min())

        seg_diff_list = np.array(seg_diff_list)
        seg_diff_list = (seg_diff_list - seg_diff_list.min()) / (seg_diff_list.max() - seg_diff_list.min())

        scores = sim_ge_list * seg_diff_list
        return np.argsort(scores)[::-1][:n]

    def get_data_9(self, unlabeled_data, n, normalize=False):
        loader = DataLoader(unlabeled_data, shuffle=False, **self.args_task['loader_te_args'])

        sim_ge_list = []
        seg_diff_list = []
        density = []
        with torch.no_grad():
            for batch_idx, (x, y, idxs) in enumerate(loader):
                x, _ = x.to(self.device), y.to(self.device)

                # 好几次取平均
                preds = []
                for _ in range(5):
                    fliped = random.random() > 0.5
                    if fliped:
                        x_h = torch.flip(x, [3])
                    else:
                        x_h = x
                    x_h = x_h + torch.randn_like(x_h) * 0.1

                    y_pred_h, seg_h = self.net.get_seg_density_predictions(x_h)
                    if fliped:
                        y_pred_h = torch.flip(y_pred_h, [3])
                    y_pred_h = y_pred_h.squeeze(0).squeeze(0)
                    preds.append(y_pred_h)

                y_pred_h = torch.mean(torch.stack(preds), dim=0)
                y_pred, seg = self.net.get_seg_density_predictions(x)
                y_pred = y_pred.squeeze(0).squeeze(0)
                seg = seg.squeeze(0).squeeze(0)

                # Calculate the similarity between the prediction and the augmented prediction
                pt_sum1 = self.get_patch_sum(y_pred, self.L_A)
                pt_sum2 = self.get_patch_sum(y_pred_h, self.L_A)

                sim_ge_list.append(torch.sum(torch.abs(pt_sum1 - pt_sum2)).item())

                # seg diff
                seg = torch.sigmoid(seg)
                y_pred_seg = y_pred * torch.sigmoid(seg)
                seg_diff = abs(y_pred_seg - y_pred).sum().item()
                seg_diff_list.append(seg_diff)
                density.append(y_pred)

        # nomarlize
        sim_ge_list = np.array(sim_ge_list)
        sim_ge_list = (sim_ge_list - sim_ge_list.min()) / (sim_ge_list.max() - sim_ge_list.min())

        seg_diff_list = np.array(seg_diff_list)
        seg_diff_list = (seg_diff_list - seg_diff_list.min()) / (seg_diff_list.max() - seg_diff_list.min())

        scores = sim_ge_list * seg_diff_list
        candidate_pools = np.argsort(scores)[::-1][:n * self.subset_ratio]
        # TODO
        print('candidate_pools', candidate_pools)

        # Select data from candidate pools according to diversity measure
        unlabeled_data_patch_sum = []

        for i in range(self.L_A + 1):
            patch_temp = []
            for c in candidate_pools:
                patch_temp.append(self.get_patch_sum(density[c], i))
            patch_temp = torch.stack(patch_temp)
            unlabeled_data_patch_sum.append(patch_temp)

        # Get dissimilarity matrix
        dissimilarity_matrix = self.calculate_all_dissimilarities(unlabeled_data_patch_sum)

        # Apply k-means
        clusters = self.kmeans_modified(dissimilarity_matrix, n)
        # TODO
        print(clusters)

        # cluster patch sum in unlabeled_data_patch_sum
        cluster_patch_sum = []

        for i in range(len(clusters)):
            cluster_idx = torch.tensor(clusters[i])
            # cluster_idx = torch.tensor([1, 2,3])
            new_unlabeled_data_patch_sum = [ps[cluster_idx] for ps in unlabeled_data_patch_sum]
            cluster_patch_sum.append(new_unlabeled_data_patch_sum)

        # calculate the similarity between the labeled data and the data in the cluster
        # Select the most dissimilar data from the labeled pool
        _, labeled_data = self.dataset.get_labeled_data()
        labeled_loader = DataLoader(labeled_data, shuffle=False, **self.args_task['loader_te_args'])
        # TODO
        np.save(f'dissimilarity{len(labeled_loader)}_matrix.npy', dissimilarity_matrix)

        labeled_data_patch_sum = []
        for i in range(self.L_A + 1):
            patch_temp = []
            for _, y, _ in labeled_loader:
                y = y.to(self.device)
                y = y.squeeze(0).squeeze(0)
                patch_temp.append(self.get_patch_sum(y, i))
            patch_temp = torch.stack(patch_temp)
            labeled_data_patch_sum.append(patch_temp)

        selected_data = []
        for i in range(n):
            cluster = cluster_patch_sum[i]
            temp_result = torch.zeros((len(cluster[0]), len(labeled_loader)), device=self.device)
            for level in range(len(cluster)):
                # iterate over all the data in the cluster
                for j in range(len(cluster[0])):
                    temp_result[j] += torch.abs(cluster[level][j] - labeled_data_patch_sum[level]).sum(
                        dim=(-1, -2)).reshape(-1)
            # 先取最小
            # temp_result = [min(temp_result[j]) for j in range(len(temp_result))]
            # temp_result = torch.stack(temp_result).cpu().numpy()
            temp_result = temp_result.min(dim=1).values.cpu().numpy()
            # TODO
            print(temp_result.shape)
            selected_data.append(clusters[i][np.argmax(temp_result)])

        # 将selected data 映射到原始的数据集
        selected_data = [candidate_pools[c] for c in selected_data]
        return np.array(selected_data)

    # 10就是在每个kmean的cluster中取一个最大的uncertainty的数据
    def get_data_10(self, unlabeled_data, n, normalize=False):
        loader = DataLoader(unlabeled_data, shuffle=False, **self.args_task['loader_te_args'])

        sim_ge_list = []
        seg_diff_list = []
        density = []
        with torch.no_grad():
            for batch_idx, (x, y, idxs) in enumerate(loader):
                x, _ = x.to(self.device), y.to(self.device)

                # 好几次取平均
                preds = []
                for _ in range(3):
                    fliped = random.random() > 0.5
                    if fliped:
                        x_h = torch.flip(x, [3])
                    else:
                        x_h = x
                    x_h = x_h + torch.randn_like(x_h) * 0.1

                    y_pred_h, seg_h = self.net.get_seg_density_predictions(x_h)
                    if fliped:
                        y_pred_h = torch.flip(y_pred_h, [3])
                    y_pred_h = y_pred_h.squeeze(0).squeeze(0)
                    preds.append(y_pred_h)

                y_pred_h = torch.mean(torch.stack(preds), dim=0)
                y_pred, seg = self.net.get_seg_density_predictions(x)
                y_pred = y_pred.squeeze(0).squeeze(0)
                seg = seg.squeeze(0).squeeze(0)

                # Calculate the similarity between the prediction and the augmented prediction
                pt_sum1 = self.get_patch_sum(y_pred, self.L_A)
                pt_sum2 = self.get_patch_sum(y_pred_h, self.L_A)

                sim_ge_list.append(torch.sum(torch.abs(pt_sum1 - pt_sum2)).item())

                # seg diff
                seg = torch.sigmoid(seg)
                y_pred_seg = y_pred * torch.sigmoid(seg)
                seg_diff = abs(y_pred_seg - y_pred).sum().item()
                seg_diff_list.append(seg_diff)
                density.append(y_pred)

        # nomarlize
        sim_ge_list = np.array(sim_ge_list)
        sim_ge_list = (sim_ge_list - sim_ge_list.min()) / (sim_ge_list.max() - sim_ge_list.min())

        seg_diff_list = np.array(seg_diff_list)
        seg_diff_list = (seg_diff_list - seg_diff_list.min()) / (seg_diff_list.max() - seg_diff_list.min())

        scores = sim_ge_list * seg_diff_list
        candidate_pools = np.argsort(scores)[::-1][:n * self.subset_ratio]

        # Select data from candidate pools according to diversity measure
        unlabeled_data_patch_sum = []

        for i in range(self.L_A + 1):
            patch_temp = []
            for c in candidate_pools:
                patch_temp.append(self.get_patch_sum(density[c], i))
            patch_temp = torch.stack(patch_temp)
            unlabeled_data_patch_sum.append(patch_temp)

        # Get dissimilarity matrix
        dissimilarity_matrix = self.calculate_all_dissimilarities(unlabeled_data_patch_sum)

        # Apply k-means
        clusters = self.kmeans_modified(dissimilarity_matrix, n)
        # print(clusters)

        selected_data = []
        for i in range(len(clusters)):
            cluster = [candidate_pools[c] for c in clusters[i]]
            # find the most uncertain data in the cluster
            uncertainty_in_cluster = [scores[c] for c in cluster]
            selected_data.append(cluster[np.argmax(uncertainty_in_cluster)])

        return np.array(selected_data)

    def get_patch_sum(self, arr, level):
        target_size = 2 ** level

        # Calculate padding size
        pad0 = ((arr.shape[0] + target_size - 1) // target_size) * target_size - arr.shape[0]
        pad1 = ((arr.shape[1] + target_size - 1) // target_size) * target_size - arr.shape[1]

        arr = torch.tensor(arr, dtype=torch.float64, device=self.device)

        # Pad array
        arr = F.pad(arr, (0, pad1, 0, pad0))

        # Calculate patch size
        patch_size0 = arr.shape[0] // target_size
        patch_size1 = arr.shape[1] // target_size

        # Create a kernel for the patch size
        kernel = torch.ones(1, 1, patch_size0, patch_size1)
        kernel = kernel.to(torch.float64)
        kernel = kernel.to(self.device)

        # Add extra dimensions for batch and channel
        # Convolve with kernel to calculate mean squared error for each patch
        arr = arr.unsqueeze(0).unsqueeze(0)

        return F.conv2d(arr, kernel, stride=(patch_size0, patch_size1), )

    def calculate_all_dissimilarities(self, data_patch_sum):
        num_images = data_patch_sum[0].shape[0]

        # matrix for dissimilarities
        dissimilarity_matrix = torch.zeros((num_images, num_images))

        # calculate the dissimilarity at different level
        for i in range(len(data_patch_sum)):
            for j in range(num_images):
                for k in range(j + 1, num_images):  # 利用 dissimilarity 的对称性，只计算一半
                    dissimilarity = torch.abs(data_patch_sum[i][j] - data_patch_sum[i][k]).sum(dim=(-1, -2))
                    # 更新 dissimilarity_matrix 中的值，如果当前 dissimilarity 更小的话
                    dissimilarity_matrix[j, k] += dissimilarity.item()
                    dissimilarity_matrix[k, j] = dissimilarity_matrix[j, k]  # 复制到下三角

        return dissimilarity_matrix

    def kcenter_greedy(self, dis_matrix, K):
        # First randomly choose a center
        centroids = []
        N = dis_matrix.shape[0]
        init_point = np.random.randint(0, N)
        centroids.append(init_point)

        for i in range(1, K):
            # Calculate the distance from point to centroids
            dis_centroid = dis_matrix[:, centroids].clone()
            # Ignore the centroids
            dis_centroid = dis_centroid.min(axis=1)[0]
            dis_centroid[centroids] = -1
            # Select maximum distance (most dissimilar)
            new_c = np.argmax(dis_centroid)
            centroids.append(new_c.item())

        return np.array(centroids)

    def kmeans(self, dis_matrix, K, n_iter=100):
        centroids = self.kcenter_greedy(dis_matrix, K)

        # Check centroid length
        if len(centroids) != K:
            raise ValueError('The number of centroids is not equal to K')

        data_idx = np.arange(dis_matrix.shape[0])

        assign_dis_records = []

        empyt_cluster = False
        for _ in range(n_iter):
            # get distance
            centroid_dis = dis_matrix[:, centroids]
            # Assign each data point to the closest centroid
            cluster_assign = np.argmin(centroid_dis, axis=1)
            # calculate the distance
            assign_dis = centroid_dis.min(axis=1)[0].sum()
            assign_dis_records.append(assign_dis)

            new_centroids = []
            for i in range(K):
                cluster_i = data_idx[cluster_assign == i]
                if len(cluster_i) == 0:
                    empyt_cluster = True
                    break
                dis_mat_i = dis_matrix[cluster_i][:, cluster_i]

                new_centroid_i = cluster_i[np.argmin(dis_mat_i.sum(axis=1))]
                new_centroids.append(new_centroid_i)
            if empyt_cluster:
                break
            centroids = np.array(new_centroids)

        return assign_dis_records, centroids.tolist()

    def kmeans_modified(self, dis_matrix, K, n_iter=100):
        centroids = self.kcenter_greedy(dis_matrix, K)

        # Check centroid length
        if len(centroids) != K:
            raise ValueError('The number of centroids is not equal to K')

        data_idx = np.arange(dis_matrix.shape[0])

        assign_dis_records = []

        empty_cluster = False
        old_centroids = centroids
        for _ in range(n_iter):
            # get distance
            centroid_dis = dis_matrix[:, centroids]
            # Assign each data point to the closest centroid
            cluster_assign = np.argmin(centroid_dis, axis=1)
            assign_dis = centroid_dis.min(axis=1)[0].sum()
            assign_dis_records.append(assign_dis)

            new_centroids = []
            for i in range(K):
                cluster_i = data_idx[cluster_assign == i]
                if len(cluster_i) == 0:
                    empty_cluster = True
                    break

                dis_mat_i = dis_matrix[cluster_i][:, cluster_i]

                new_centroid_i = cluster_i[np.argmin(dis_mat_i.sum(axis=1))]
                new_centroids.append(new_centroid_i)

            if empty_cluster:
                break
            old_centroids = centroids
            centroids = np.array(new_centroids)

        # get the cluster assignment, i.e. the index of the cluster each data point belongs to
        centroid_dis = dis_matrix[:, centroids]
        cluster_assign = np.argmin(centroid_dis, axis=1)

        # check the number of clusters
        for i in range(K):
            cluster_i = data_idx[cluster_assign == i]
            if len(cluster_i) == 0:
                centroids = old_centroids
                # Recalculate the cluster assignment
                centroid_dis = dis_matrix[:, centroids]
                cluster_assign = np.argmin(centroid_dis, axis=1)
                break

        # Group the data points by cluster assignment
        clusters = []
        for i in range(K):
            cluster_i = data_idx[cluster_assign == i]
            clusters.append(cluster_i)

        return clusters
