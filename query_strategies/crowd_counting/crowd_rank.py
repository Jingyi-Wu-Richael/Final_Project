from ..strategy import Strategy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random


class CrowdRank(Strategy):
    def __init__(self, dataset, net, args_input, args_task):
        super(CrowdRank, self).__init__(dataset, net, args_input, args_task)
        self.args_input = args_input
        self.args_task = args_task
        self.dataset = dataset
        self.net = net
        # Number of times calling generate_ranked_images for each image
        self.K = 10

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        image_uncertainty = self.get_image_rank_score(unlabeled_data)
        return unlabeled_idxs[image_uncertainty.sort()[1][:n]]

    def get_image_rank_score(self, unlabeled_data):
        image_rank_score = torch.zeros(len(unlabeled_data))
        loader_data = DataLoader(unlabeled_data, shuffle=False, **self.args_task['loader_te_args'])

        for batch_idx, (inputs, targets, idx) in enumerate(loader_data):
            score = 0
            count = 0
            for i in range(self.K):
                ranked_images = self.generate_ranked_images(inputs)
                ranked_predictions = self.net.predict_single(ranked_images)
                ranked_predictions = [ranked.sum() for ranked in ranked_predictions]
                # Calculate the uncertainty
                img_score, pair_count = self.calculate_ranking_score(ranked_predictions)
                score += img_score
                count += pair_count
            image_rank_score[batch_idx] = score / count

        return image_rank_score

    def calculate_ranking_score(self, imgs):
        score = 0
        count = 0
        len_img = len(imgs)
        for i in range(len_img):
            for j in range(i + 1, len_img):
                count += 1
                if imgs[i] > imgs[j]:
                    score += 1
        return score, count

    def generate_ranked_images(self, img, r=8, s=0.75, k=5):
        """Generate the ranked images for the given image.

        Args:
            img (torch.Tensor): Image to be processed
            r: Anchor region size
            s: Scale factor
            k: Number of ranked images to be generated

        Returns:
            List of ranked image patches.
        """
        # Find anchor point
        if len(img.shape) == 5:
            img = img.squeeze(0)
        width, height = img.shape[2:]
        anchor_width, anchor_height = width // r, height // r
        anchor_top_left = width // 2 - anchor_width // 2, height // 2 - anchor_height // 2
        anchor_point = random.randint(0, anchor_width), random.randint(0, anchor_height)
        anchor_point = anchor_top_left[0] + anchor_point[0], anchor_top_left[1] + anchor_point[1]

        # Largest square patch centered at anchor point
        patch_width, patch_height = min(anchor_point[0], width - anchor_point[0]), min(anchor_point[1],
                                                                                       height - anchor_point[1])
        patch_size = min(patch_width, patch_height)
        largest_size = patch_size * 2
        largest_patch = img[:, :, anchor_point[0] - patch_size:anchor_point[0] + patch_size,
                        anchor_point[1] - patch_size:anchor_point[1] + patch_size]
        # Generate ranked images
        ranked_images = [largest_patch]
        for i in range(k - 1):
            patch_size = int(patch_size * s)
            patch = img[:, :, anchor_point[0] - patch_size:anchor_point[0] + patch_size,
                    anchor_point[1] - patch_size:anchor_point[1] + patch_size]
            ranked_images.append(patch)

        # Collate ranked images into a mini-batch through resize
        ranked_images = torch.cat([F.interpolate(img, size=(largest_size, largest_size)) for img in ranked_images],
                                  dim=0)

        return ranked_images
