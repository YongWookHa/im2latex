import torch
import torch.nn as nn

from utils.utils import add_positional_features

class Encoder(nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.cnn = CNN()

        # transform (H' -> 1)
        # self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))

        ## to use all grids
        self.unfold = nn.Unfold(1)

        # self.bi_lstm = BidirectionalLSTM(512, cfg.enc_hidden_size, cfg.vocab_size)
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

    def forward(self, images):

        return self.cnn(images)  # [B, W', 512]

class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, num_layers=2, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, inp):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()

        recurrent, _ = self.rnn(inp)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output
