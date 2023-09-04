import torch
from torch import nn


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