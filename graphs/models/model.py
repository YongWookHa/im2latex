import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform

from graphs.weights_initializer import weights_init

class Im2LatexModel(nn.Module):
    def __init__(self, cfg, out_size):
        super(Im2LatexModel, self).__init__()
        self.cfg = cfg
        enc_out_dim = cfg.enc_out_dim
        emb_dim = cfg.emb_dim
        dec_rnn_h = cfg.dec_rnn_h

        # define layers
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.Tanh(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.Tanh(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.Tanh(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(256, cfg.enc_out_dim, 3, 1, 1),
            nn.Tanh(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(enc_out_dim, enc_out_dim, 3, 1, 1),
            nn.Tanh()
        )

        # need to be tested
        self.LSTMCell_1 = nn.LSTMCell(emb_dim + dec_rnn_h, dec_rnn_h)
        self.LSTMCell_2 = nn.LSTMCell(emb_dim + dec_rnn_h, dec_rnn_h)

        self.embedding = nn.Embedding(out_size, emb_dim)

        self.init_wh = nn.Linear(enc_out_dim, dec_rnn_h)
        self.init_wc = nn.Linear(enc_out_dim, dec_rnn_h)
        self.init_wo = nn.Linear(enc_out_dim, dec_rnn_h)

        #Attention mechanism
        self.beta = nn.Parameter(torch.Tensor(enc_out_dim))
        nn.init.uniform_(sefl.beta, -0.01, 0.01)
        self.W_1 = nn.Linear(enc_out_dim, enc_out_dim, bias=False)
        self.W_2 = nn.Linear(dec_rnn_h, enc_out_dim, bias=False)
        self.W_3 = nn.Linear(dec_rnn_h + enc_out_dim, dec_rnn_h, bias=False)
        self.W_out = nn.Linear(dec_rnn_h, out_size, bias=False)

        self.uniform = Uniform(0, 1)

    def forward(self, img, formulas, epsilon=1.):
        x = self.conv(x)
        x = self.relu(x)

        out = x.view(x.size(0), -1)
        return out
