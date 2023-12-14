from typing import List

import torch
from torch import Tensor
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two same convolution layers"""
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """U-Net model for segmentation.
    The size of the input image must be divisible by 16.

    Attributes:
        in_channels (int): Number of channels of the input image
        out_channels (int): Number of classes for segmentation
        features (list[int]): Sizes of interim activation maps
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: List[int] = [32, 64, 128, 256],
    ) -> None:

        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Downsample
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Upsample
        for feature in features[::-1]:
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, 2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        # Middle part
        self.middle = DoubleConv(features[-1], features[-1] * 2)

        # Final conv
        self.conv_classifier = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.middle(x)
        skip_connections = skip_connections[::-1]

        for i, up in enumerate(self.ups):
            if i % 2 == 0:
                x = up(x)
                x = torch.cat((skip_connections[i // 2], x), dim=1)
            else:
                x = up(x)

        # no need for activation if we use BCEWithLogitsLoss
        return self.conv_classifier(x)