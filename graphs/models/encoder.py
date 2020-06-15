import torch
import torch.nn as nn

from utils.utils import add_positional_features

class CNNEncoder(nn.Module):
    def __init__(self, cfg):
        super(CNNEncoder, self).__init__()
        self.cfg = cfg

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 1),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 1),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((1, 2), (1, 2), 0),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.MaxPool2d((2, 1), (2, 1), 0),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        ## transform (H' -> 1)
        # self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))

        # to use all grids
        self.unold = nn.Unfold(1)

    def forward(self, images):
        encoded_imgs = self.cnn(images)  # [B, 512, H', W']
        # encoded_imgs = self.AdaptiveAvgPool2(encoded_imgs.permute(0, 3, 1, 2))
        memoryBank = self.unfold(encoded_imgs)  # [B, 512, H' x W']
        memoryBank += add_positional_features(memoryBank.permute(0, 2, 1))

        return memoryBank
