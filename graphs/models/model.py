'''
implementation of "Translating Math Formula Images to LaTeX Sequences Using
Deep Neural Networks with Sequence-level Training"
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import add_positional_features

class Im2LatexModel(nn.Module):
    def __init__(self, cfg, out_size):
        """
        out_size : VOCAB_SIZE
        """
        super(Im2LatexModel, self).__init__()
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available()
                                          else 'cpu')
        self.out_size = out_size
        self.lstm_hidden_size = cfg.lstm_hidden_size
        enc_dim = cfg.enc_dim
        emb_dim = cfg.emb_dim
        attn_dim = out_size if cfg.attn_dim == "vocab_size" else cfg.attn_dim

        # define layers
        self.cnn_encoder = nn.Sequential(
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
        self.unfold = nn.Unfold(1)
        self.embedding = nn.Embedding(out_size, emb_dim)

        # self.init_wh1 = nn.Linear(512, lstm_hidden_size)
        # self.init_wc1 = nn.Linear(512, lstm_hidden_size)
        # self.init_wh2 = nn.Linear(512, lstm_hidden_size)
        # self.init_wc2 = nn.Linear(512, lstm_hidden_size)
        # self.init_o = nn.Linear(512, lstm_hidden_size)

        self.DeepOutputLayer = nn.Linear(
                        self.lstm_hidden_size + enc_dim + emb_dim, out_size)

        self.attn_wh = nn.Linear(self.lstm_hidden_size, attn_dim)
        self.attn_wa = nn.Linear(enc_dim, attn_dim)
        self.attn_combine = nn.Linear(attn_dim, 1)

        self.LSTM = nn.LSTM(emb_dim+enc_dim, hidden_size=self.lstm_hidden_size,
                                                num_layers=2, batch_first=True)


    def forward(self, imgs, formulas=None, is_train=False):
        """
        imgs: [B, C, H, W]
        formulas: [B, max_seq_len]
        returns: logit of [B, MAX_LEN, VOCAB_SIZE]
        """
        max_seq_len = self.cfg.max_len

        # encoding
        memoryBank = self.encode(imgs)  # [B, 512, H', W']

        logits = []
        bos = 0
        bs = imgs.size(0)

        if not is_train:
            logit = torch.zeros((bs, self.out_size))
            logit[:, 0] = torch.tensor(1)
            logit = logit.unsqueeze(1).to(self.device)
        prev_h, dec_states = self.init_decoder(memoryBank)

        for i in range(max_seq_len):
            if is_train:
                tgt = formulas[:, i]
            else:
                tgt = torch.argmax(logit, dim=2).squeeze(1)
            prev_w = self.embedding(tgt).unsqueeze(1)
            z_t = self.get_soft_attn(prev_h, memoryBank).unsqueeze(1)  # [B, 1, D]
            self.LSTM.flatten_parameters()
            h_t, dec_states =  self.LSTM(torch.cat([z_t, prev_w], dim=2), dec_states)
            logit = self.DeepOutputLayer(torch.cat([h_t, z_t, prev_w], dim=2))

            logits.append(F.log_softmax(logit, dim=2))
            prev_h = h_t

        return torch.stack(logits).permute(1,0,2,3).squeeze(2)

    def encode(self, imgs):
        encoded_imgs = self.cnn_encoder(imgs)  # [B, 512, H', W']
        memoryBank = self.unfold(encoded_imgs)  # [B, 512, H'*W']
        memoryBank = memoryBank.permute(0,2,1)  # [B, H'*W', 512]  # L = H'*W'

        memoryBank += add_positional_features(memoryBank)
        return memoryBank

    def get_soft_attn(self, prev_h, memoryBank):
        '''
        prev_h : [B, 1, D]
        memoryBank : [B, L, enc_dim]
        return :
        '''
        # Attention
        att_h = self.attn_wh(prev_h)
        att_m = self.attn_wa(memoryBank)

        att = self.attn_combine(torch.tanh(att_h+att_m))
        attn_weight = F.softmax(att, dim=1)  # alpha: [B, L]

        z = torch.sum(torch.mul(attn_weight, memoryBank), dim=1)  # [B, L]

        return z

    def init_decoder(self, memoryBank):
        bs = memoryBank.size(0)
        o = torch.zeros((bs, 1, self.lstm_hidden_size)).to(self.device)
        h = c = torch.zeros((2, bs, self.lstm_hidden_size)).to(self.device)

        return o, (h, c)

#    def init_decoder(self, memoryBank):
#        mean_enc_out = memoryBank.mean(dim=1)
#        h1 = torch.tanh(self.init_wh1(mean_enc_out))
#        c1 = torch.tanh(self.init_wc1(mean_enc_out))
#        h2 = torch.tanh(self.init_wh2(mean_enc_out))
#        c2 = torch.tanh(self.init_wc2(mean_enc_out))
#        o = torch.tanh(self.init_o(mean_enc_out)).unsqueeze(1)
#
#        return o, (torch.stack([h1, h2]), torch.stack([c1, c2]))
