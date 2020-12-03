import torch
import torch.nn as nn

from utils.utils import add_positional_features

class Encoder(nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.cnn = CNN()

        ## to use all grids
        self.unfold = nn.Unfold(1)
        self.linear = nn.Linear(512, cfg.vocab_size)

    def forward(self, images):
        """ Feature Extraction """
        encoded_imgs = self.cnn(images)  # [B, 512, H', W']
        encoded_imgs = self.unfold(encoded_imgs).permute(0,2,1)  # [B, L=W'*H', 512]
        encoded_imgs += add_positional_features(encoded_imgs)  # [B, L, 512]

        """ Sequence Modeling """
        contextual_features = self.linear(encoded_imgs)

        return contextual_features


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2, 1),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2, 1),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d((1, 2), (1, 2), 0),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.MaxPool2d((2, 1), (2, 1), 0),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(True)
        )

    def forward(self, images):
        return self.cnn(images)  # [B, W', 512]
