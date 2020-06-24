'''
implementation of "Translating Math Formula Images to LaTeX Sequences Using
Deep Neural Networks with Sequence-level Training"
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from graphs.models.encoder import Encoder
from graphs.models.decoder import Decoder

class Im2LatexModel(nn.Module):
    def __init__(self, cfg):
        """
        out_size : VOCAB_SIZE
        """
        self.max_len = cfg.max_len
        super(Im2LatexModel, self).__init__()

        """ Define Layers """
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)


    def forward(self, imgs, formulas=None, is_train=False):
        """
        input
            imgs: [B, C, H, W]
            formulas: [B, max_seq_len]
        return
            logit of [B, MAX_LEN, VOCAB_SIZE]
        """

        """ encode """
        contextual_features = self.encoder(imgs)  # [B, W', D]
        """ decode """
        prediction = self.decoder(contextual_features.contiguous(), formulas,
                        is_train, batch_max_length=self.max_len)

        return prediction
