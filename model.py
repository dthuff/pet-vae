"""
Architecture adapted from https://github.com/duyphuongcri/Variational-AutoEncoder
"""

import torch
import torch.nn as nn
import math


class ConvBlock(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class ResNetBlock(nn.Module):
    """
    ResNet block - two blocks of sequential conv, batchnorm, relu
    """

    def __init__(self, channels, kernel_size, stride=1, padding=1):
        super(ResNetBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.conv(x) + x
        return out


class UpConvBlock(nn.Module):
    """
    UpConv block - conv with 1x1 kernel and upsample to recover spatial dims in decoder
    """

    def __init__(self, channels_in, channels_out, kernel_size=1, scale_factor=2, align_corners=False):
        super(UpConvBlock, self).__init__()
        self.up = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size),
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=align_corners),
        )

    def forward(self, x):
        return self.up(x)


class Encoder(nn.Module):
    """
    Class for the Encoder (1st half of the VAE)
    4 blocks of conv, resnet, maxpool

    So, at bottleneck layer, input spatial dims are reduced by factor of 2^4
    """

    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = ConvBlock(channels_in=1, channels_out=32, kernel_size=3)
        self.res_block1 = ResNetBlock(channels=32, kernel_size=3)
        self.MaxPool1 = nn.MaxPool2d(3, stride=2, padding=1)

        self.conv2 = ConvBlock(channels_in=32, channels_out=64, kernel_size=3)
        self.res_block2 = ResNetBlock(channels=64, kernel_size=3)
        self.MaxPool2 = nn.MaxPool2d(3, stride=2, padding=1)

        self.conv3 = ConvBlock(channels_in=64, channels_out=128, kernel_size=3)
        self.res_block3 = ResNetBlock(channels=128, kernel_size=3)
        self.MaxPool3 = nn.MaxPool2d(3, stride=2, padding=1)

        self.conv4 = ConvBlock(channels_in=128, channels_out=256, kernel_size=3)
        self.res_block4 = ResNetBlock(channels=256, kernel_size=3)
        self.MaxPool4 = nn.MaxPool2d(3, stride=2, padding=1)

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            stdv = 1.0 / math.sqrt(weight.size(0))
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.res_block1(x1)
        x1 = self.MaxPool1(x1)

        x2 = self.conv2(x1)
        x2 = self.res_block2(x2)
        x2 = self.MaxPool2(x2)

        x3 = self.conv3(x2)
        x3 = self.res_block3(x3)
        x3 = self.MaxPool3(x3)

        x4 = self.conv4(x3)
        x4 = self.res_block4(x4)
        x4 = self.MaxPool4(x4)
        return x4


class Decoder(nn.Module):
    """
    Class for the decoder half of the VAE
    """
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.linear_up = nn.Linear(latent_dim, 16384)  # 2nd arg is 256*8
        self.relu = nn.ReLU()
        self.upsize4 = UpConvBlock(channels_in=256, channels_out=128, kernel_size=1, scale_factor=2)
        self.res_block4 = ResNetBlock(channels=128, kernel_size=3)
        self.upsize3 = UpConvBlock(channels_in=128, channels_out=64, kernel_size=1, scale_factor=2)
        self.res_block3 = ResNetBlock(channels=64, kernel_size=3)
        self.upsize2 = UpConvBlock(channels_in=64, channels_out=32, kernel_size=1, scale_factor=2)
        self.res_block2 = ResNetBlock(channels=32, kernel_size=3)
        self.upsize1 = UpConvBlock(channels_in=32, channels_out=1, kernel_size=1, scale_factor=2)
        self.res_block1 = ResNetBlock(channels=1, kernel_size=3)

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            stdv = 1.0 / math.sqrt(weight.size(0))
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x):
        x4_ = self.linear_up(x)
        x4_ = self.relu(x4_)

        # x4_ = x4_.view(-1, 256, 5, 6, 5)
        x4_ = x4_.view(-1, 256, 8, 8)
        x4_ = self.upsize4(x4_)
        x4_ = self.res_block4(x4_)

        x3_ = self.upsize3(x4_)
        x3_ = self.res_block3(x3_)

        x2_ = self.upsize2(x3_)
        x2_ = self.res_block2(x2_)

        x1_ = self.upsize1(x2_)
        x1_ = self.res_block1(x1_)

        return x1_


class VAE(nn.Module):
    """
    Variational autoencoder consists of encoder + decoder
    """
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.latent_dim = latent_dim
        self.z_mean = nn.Linear(16384, latent_dim)  # First arg was 256*150
        self.z_log_sigma = nn.Linear(16384, latent_dim)
        self.epsilon = torch.normal(size=(1, latent_dim), mean=0, std=1.0, device=self.device)
        self.encoder = Encoder()
        self.decoder = Decoder(latent_dim)

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            stdv = 1.0 / math.sqrt(weight.size(0))
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        z_mean = self.z_mean(x)
        z_log_sigma = self.z_log_sigma(x)
        z = z_mean + z_log_sigma.exp() * self.epsilon
        y = self.decoder(z)
        return y, z_mean, z_log_sigma
