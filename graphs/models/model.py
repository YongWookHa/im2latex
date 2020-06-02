'''
implementation of "Translating Math Formula Images to LaTeX Sequences Using
Deep Neural Networks with Sequence-level Training"
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

# from utils.utils import add_positional_features

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
        


        self.attn = nn.Linear(512, 512)
        self.attn_combine = nn.Linear(512+512, 512)

        self.LSTM = nn.LSTM(emb_dim+512, hidden_size=512, num_layers=2,
                              bidirectional=False, batch_first=True)


    def forward(self, imgs, formulas):
        """
        imgs: [B, C, H, W]
        formulas: [B, max_seq_len]
        returns: logit of [B, MAX_LEN, VOCAB_SIZE]
        """
        # encoding
        memoryBank = self.encode(imgs)  # [B, 512, H', W']

        max_seq_len = formulas.size(1)
        dec_states = self.init_decoder(memoryBank)

        # LSTM input: (inp, (h_n, c_n))
        # inp : (seq_len, batch, input_size)

        logits = self.decode(dec_states, formulas, memoryBank, max_seq_len)

        return torch.stack(logits).permute(1,0,2,3).squeeze(2)

    def encode(self, imgs):
        encoded_imgs = self.cnn_encoder(imgs)  # [B, 512, H', W']
        memoryBank = self.unfold(encoded_imgs)  # [B, 512, H'*W']
        memoryBank = memoryBank.permute(0,2,1)  # [B, H'*W', 512]  # L = H'*W'

        memoryBank += add_positional_features(memoryBank)
        return memoryBank

    def decode(self, dec_states, formulas, memoryBank, max_seq_len):
        # step decoding
        logits = []
        prev_h, _ = dec_states  # [B, 1, D]
        for i in range(max_seq_len):
            tgt = formulas[:, i]
            prev_w = self.embedding(tgt).unsqueeze(1)
            C_t = self.get_attn(prev_h, memoryBank).unsqueeze(1)  # [B, 1, D]
            h_t, dec_states =  self.LSTM(torch.cat([prev_w, prev_h], dim=2), dec_states)
            O_t = torch.tanh(self.dec_W3(torch.cat([h_t, C_t], dim=2)))  # O_t = [B, 1, D]
            logit = F.softmax(self.dec_W4(O_t), dim=2)
            logits.append(logit)
            
            prev_h = h_t
        
        return logits    

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
        # o = torch.tanh(self.init_o(mean_enc_out)).unsqueeze(1)

        return (torch.stack([h1, h2]), torch.stack([c1, c2]))
        #return o, (torch.stack([h1, h2]), torch.stack([c1, c2]))


def add_positional_features(tensor: torch.Tensor,
                            min_timescale: float = 1.0,
                            max_timescale: float = 1.0e4):
    """
    Implements the frequency-based positional encoding described
    in `Attention is all you Need
    Parameters
    ----------
    tensor : ``torch.Tensor``
        a Tensor with shape (batch_size, timesteps, hidden_dim).
    min_timescale : ``float``, optional (default = 1.0)
        The largest timescale to use.
    Returns
    -------
    The input tensor augmented with the sinusoidal frequencies.
    """
    _, timesteps, hidden_dim = tensor.size()

    timestep_range = get_range_vector(timesteps, tensor.device).data.float()
    # We're generating both cos and sin frequencies,
    # so half for each.
    num_timescales = hidden_dim // 2
    timescale_range = get_range_vector(
        num_timescales, tensor.device).data.float()

    log_timescale_increments = math.log(
        float(max_timescale) / float(min_timescale)) / float(num_timescales - 1)
    inverse_timescales = min_timescale * \
        torch.exp(timescale_range * -log_timescale_increments)

    # Broadcasted multiplication - shape (timesteps, num_timescales)
    scaled_time = timestep_range.unsqueeze(1) * inverse_timescales.unsqueeze(0)
    # shape (timesteps, 2 * num_timescales)
    sinusoids = torch.randn(
        scaled_time.size(0), 2*scaled_time.size(1), device=tensor.device)
    sinusoids[:, ::2] = torch.sin(scaled_time)
    sinusoids[:, 1::2] = torch.cos(scaled_time)
    if hidden_dim % 2 != 0:
        # if the number of dimensions is odd, the cos and sin
        # timescales had size (hidden_dim - 1) / 2, so we need
        # to add a row of zeros to make up the difference.
        sinusoids = torch.cat(
            [sinusoids, sinusoids.new_zeros(timesteps, 1)], 1)
    return tensor + sinusoids.unsqueeze(0)

def get_range_vector(size: int, device) -> torch.Tensor:
    return torch.arange(0, size, dtype=torch.long, device=device)


if __name__ == "__main__":
    from easydict import EasyDict
    import math
    cfg = EasyDict({'emb_dim':80, 'dec_rnn_h':512})
    model = Im2LatexModel(cfg, 336)
    """
    Forward parameters
    imgs: [B, C, H, W]
    formulas: [B, max_seq_len]
    returns: logit of [B, MAX_LEN, VOCAB_SIZE]
    """
    B, C, H, W = 4, 1, 1088, 128
    max_seq_len = 150
    print(model(torch.randn(B,C,H,W), torch.randint(0, 336, (B, max_seq_len))))