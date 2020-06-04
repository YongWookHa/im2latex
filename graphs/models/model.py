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
        emb_dim = cfg.emb_dim
        dec_rnn_h = cfg.dec_rnn_h

        # define layers
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=0, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=0, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), padding=0, stride=(1, 2)),

            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), padding=0, stride=(2, 1)),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512)
        )
        self.unfold = nn.Unfold(1)
        self.embedding = nn.Embedding(out_size, emb_dim)

        self.init_wh1 = nn.Linear(512, dec_rnn_h)
        self.init_wc1 = nn.Linear(512, dec_rnn_h)
        self.init_wh2 = nn.Linear(512, dec_rnn_h)
        self.init_wc2 = nn.Linear(512, dec_rnn_h)
        self.init_o = nn.Linear(512, dec_rnn_h)

        self.attn_W1 = nn.Linear(dec_rnn_h, 512, bias=False)
        self.attn_W2 = nn.Linear(512, 512, bias=False)

        self.dec_W3 = nn.Linear(dec_rnn_h*2, dec_rnn_h, bias=False)
        self.dec_W4 = nn.Linear(dec_rnn_h, out_size, bias=False)

        self.attn = nn.Linear(512, 512)
        self.attn_combine = nn.Linear(512+512, 512)

        self.LSTM = nn.LSTM(emb_dim+512, hidden_size=512, num_layers=2,
                              bidirectional=False, batch_first=True)


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
        logit = torch.zeros((imgs.size(0), max_seq_len))
        prev_h, dec_states = self.init_decoder(memoryBank)

        for i in range(max_seq_len):
            if is_train:
                tgt = formulas[:, i]
            else:
                tgt = torch.argmax(logit, dim=2).squeeze(1)
            prev_w = self.embedding(tgt).unsqueeze(1)
            C_t = self.get_attn(prev_h, memoryBank).unsqueeze(1)  # [B, 1, D]
            self.LSTM.flatten_parameters()
            h_t, dec_states =  self.LSTM(torch.cat([prev_w, prev_h], dim=2), dec_states)
            O_t = torch.tanh(self.dec_W3(torch.cat([h_t, C_t], dim=2)))  # O_t = [B, 1, D]
            logit = F.log_softmax(self.dec_W4(O_t), dim=2)
            logits.append(logit)

            prev_h = h_t

        return torch.stack(logits).permute(1,0,2,3).squeeze(2)

    def encode(self, imgs):
        encoded_imgs = self.cnn_encoder(imgs)  # [B, 512, H', W']
        memoryBank = self.unfold(encoded_imgs)  # [B, 512, H'*W']
        memoryBank = memoryBank.permute(0,2,1)  # [B, H'*W', 512]  # L = H'*W'

        memoryBank += add_positional_features(memoryBank)
        return memoryBank

    def get_attn(self, prev_h, memoryBank):
        '''
        prev_h : [B, 1, D]
        memoryBank : [B, L, D]
        '''
        # Attention
        a = torch.tanh(torch.add(self.attn_W1(prev_h),
                                 self.attn_W2(memoryBank)))
        attn_weight = F.softmax(a, dim=2)  # alpha: [B, L]

        C_t = torch.sum(torch.mul(attn_weight, memoryBank), dim=1)  # [B, D]

        return C_t


    def init_decoder(self, memoryBank):
        mean_enc_out = memoryBank.mean(dim=1)
        h1 = torch.tanh(self.init_wh1(mean_enc_out))
        c1 = torch.tanh(self.init_wc1(mean_enc_out))
        h2 = torch.tanh(self.init_wh2(mean_enc_out))
        c2 = torch.tanh(self.init_wc2(mean_enc_out))
        o = torch.tanh(self.init_o(mean_enc_out)).unsqueeze(1)

        return o, (torch.stack([h1, h2]), torch.stack([c1, c2]))
