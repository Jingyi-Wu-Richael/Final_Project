import numpy as np
from torch.nn import init
from torch.utils.data import DataLoader
from tqdm import tqdm

from .sampler import Sampler
import torch.nn as nn
import torch
from handlers import ShanghaitechHandler_AE

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class AutoEncoderSampler(Sampler):
    def __init__(self, dataset, args_input, args_task):
        super().__init__(dataset, args_input, args_task)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = 5e-4
        self.epoch = 50
        self.batch_size = 10

    def sample(self, n):
        print("Sampling {} points using AutoEncoderSampler".format(n))
        # Get unlabeled data, (Should expect to be all the data)
        X, _ = self.dataset.get_partial_unlabeled_data()

        latent_vectors = self.get_latent_vectors(X)

        new_data_points = self.k_center_greery(latent_vectors, n)

        # Update dataset
        self.dataset.labeled_idxs[new_data_points] = True

    def k_center_greery(self, latents, n):
        labeled_idxs, _ = self.dataset.get_train_data()

        pca = PCA(n_components=50)
        latents = pca.fit_transform(latents)
        latents = latents.astype(np.float16)

        dist_mat = np.matmul(latents, latents.transpose())
        sq = np.array(dist_mat.diagonal()).reshape(len(labeled_idxs), 1)
        dist_mat *= -2
        dist_mat += sq
        dist_mat += sq.transpose()
        dist_mat = np.sqrt(dist_mat)

        # labeled_idxs should be empty, so randomly select a center
        if not np.any(labeled_idxs):
            # If it is, randomly select an index to be the first center
            first_center = np.random.choice(self.dataset.n_pool)
            labeled_idxs[first_center] = True
            iter_count = n - 1
        else:
            iter_count = n

        mat = dist_mat[~labeled_idxs, :][:, labeled_idxs]

        for i in tqdm(range(iter_count), ncols=100):
            mat_min = mat.min(axis=1)
            q_idx_ = mat_min.argmax()
            q_idx = np.arange(self.dataset.n_pool)[~labeled_idxs][q_idx_]
            labeled_idxs[q_idx] = True
            mat = np.delete(mat, q_idx_, 0)
            mat = np.append(mat, dist_mat[~labeled_idxs, q_idx][:, None], axis=1)

        return np.arange(self.dataset.n_pool)[(self.dataset.labeled_idxs ^ labeled_idxs)]

    def get_latent_vectors(self, data):
        unlabeled_loader = DataLoader(ShanghaitechHandler_AE(data, transform=self.args_task['transform_train']),
                                      shuffle=True,
                                      batch_size=self.batch_size)
        mse_loss = nn.MSELoss()
        net = DenoisingAE()
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        net.to(self.device)
        net.train()

        pbar = tqdm(range(self.epoch), desc="Training")
        for epoch in pbar:
            for img, _ in unlabeled_loader:
                img = img.to(self.device)
                opt.zero_grad()
                recon, z = net(img)
                loss = mse_loss(img, recon)
                loss.backward()
                opt.step()
        pbar.close()

        latent_vectors = []
        unlabeled_loader = DataLoader(ShanghaitechHandler_AE(data, transform=self.args_task['transform_train']),
                                      shuffle=False,
                                      batch_size=1)
        for img, _ in unlabeled_loader:
            img = img.to(self.device)
            _, z = net(img)
            z = nn.functional.adaptive_avg_pool2d(z, (1, 1)).view(-1)
            latent_vectors.append(z.cpu().detach().numpy())

        return np.array(latent_vectors)


class EncoderBlock(nn.Module):
    def __init__(self, filter_in, filter_out, kernel, stride, padding):
        super(EncoderBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(filter_in, filter_out, kernel, stride, padding, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2),
            nn.BatchNorm2d(filter_out),
        )

    def forward(self, x):
        return self.layer(x)


class DecoderBlock(nn.Module):
    def __init__(self, filter_in, filter_out, kernel, stride, padding):
        super(DecoderBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(filter_in, filter_out, kernel, stride, padding, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2),
            nn.BatchNorm2d(filter_out),
        )

    def forward(self, x):
        return self.layer(x)


class DenoisingAE(nn.Module):
    def __init__(self):
        super(DenoisingAE, self).__init__()
        # 256 * 256
        self.encoder = nn.Sequential(
            EncoderBlock(3, 64, 3, 1, 1),
            EncoderBlock(64, 64, 4, 2, 1),  # 128 * 128
            EncoderBlock(64, 128, 4, 2, 1),  # 64 * 64
            EncoderBlock(128, 256, 4, 2, 1),  # 32 * 32
            EncoderBlock(256, 256, 4, 2, 1),  # 16 * 16
            EncoderBlock(256, 512, 4, 2, 1),  # 8 * 8
        )
        self.decoder = nn.Sequential(
            DecoderBlock(512, 256, 4, 2, 1),  # 16 * 16
            DecoderBlock(256, 256, 4, 2, 1),  # 32 * 32
            DecoderBlock(256, 128, 4, 2, 1),  # 64 * 64
            DecoderBlock(128, 64, 4, 2, 1),  # 128 * 128
            DecoderBlock(64, 64, 4, 2, 1),  # 256 * 256
            nn.Conv2d(64, 3, 3, 1, 1),
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z
# class AE(nn.Module):
#     def __init__(self, z_dim=256, nc=3):
#         super(AE, self).__init__()
#         self.z_dim = z_dim
#         self.nc = nc
#         self.encoder = nn.Sequential(
#             nn.Conv2d(nc, 32, 4, 2, 1, bias=True),  # B,  32, 256
#             nn.BatchNorm2d(32),
#             nn.ReLU(True),
#             nn.Conv2d(32, 64, 4, 2, 1, bias=True),  # B,  64, 128
#             nn.BatchNorm2d(64),
#             nn.ReLU(True),
#             nn.Conv2d(64, 128, 4, 2, 1, bias=True),  # B,  128, 64
#             nn.BatchNorm2d(128),
#             nn.ReLU(True),
#             nn.Conv2d(128, 256, 4, 2, 1, bias=True),  # B,  256, 32
#             nn.BatchNorm2d(256),
#             nn.ReLU(True),
#             nn.Conv2d(256, 512, 4, 2, 1, bias=True),  # B,  512,  16
#             nn.BatchNorm2d(512),
#             nn.ReLU(True),
#             nn.Conv2d(512, 1024, 4, 2, 1, bias=True),  # B, 1024, 8, 8
#             nn.BatchNorm2d(1024),
#             nn.ReLU(True),
#             View((-1, 1024 * 8 * 8)),  # B, 1024*4*4
#         )
#
#         self.fc = nn.Linear(1024 * 8 * 8, z_dim)  # B, z_dim
#         self.decoder = nn.Sequential(
#             nn.Linear(z_dim, 1024 * 8 * 8),  # B, 1024*8*8
#             View((-1, 1024, 8, 8)),  # B, 1024,  8,  8
#             nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=True),  # B,  512, 16
#             nn.BatchNorm2d(512),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=True),  # B,  256, 32
#             nn.BatchNorm2d(256),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=True),  # B,  128,  64
#             nn.BatchNorm2d(128),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=True),  # B,  64, 128
#             nn.BatchNorm2d(64),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=True),  # B,  32,  256
#             nn.BatchNorm2d(32),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=True),  # B,  32,  512
#             nn.BatchNorm2d(16),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(16, nc, 1),  # B,   nc, 256, 256
#         )
#         self.weight_init()
#
#     def weight_init(self):
#         for block in self._modules:
#             try:
#                 for m in self._modules[block]:
#                     kaiming_init(m)
#             except:
#                 kaiming_init(block)
#
#     def forward(self, x):
#         z = self._encode(x)
#         x_recon = self._decode(z)
#         return x_recon, z
#
#     def _encode(self, x):
#         return self.fc(self.encoder(x))
#
#     def _decode(self, z):
#         return self.decoder(z)
#
#
# class View(nn.Module):
#     def __init__(self, size):
#         super(View, self).__init__()
#         self.size = size
#
#     def forward(self, tensor):
#         return tensor.view(self.size)
#

# def kaiming_init(m):
#     if isinstance(m, (nn.Linear, nn.Conv2d)):
#         init.kaiming_normal_(m.weight)
#         if m.bias is not None:
#             m.bias.data.fill_(0)
#     elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
#         m.weight.data.fill_(1)
#         if m.bias is not None:
#             m.bias.data.fill_(0)
