import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import copy, deepcopy


class Decoder(nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        self.cfg = cfg
        self.attention_cell = AttentionCell(cfg.vocab_size,
                                            cfg.dec_hidden_size,
                                            cfg.vocab_size)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.hidden_size = cfg.dec_hidden_size
        self.num_classes = cfg.vocab_size
        self.generator = nn.Linear(cfg.dec_hidden_size, cfg.vocab_size)

    def _char_to_onehot(self, input_char, onehot_dim=38):
        input_char = input_char.unsqueeze(1)
        batch_size = input_char.size(0)
        one_hot = torch.FloatTensor(batch_size, onehot_dim).zero_().to(self.device)
        one_hot = one_hot.scatter_(1, input_char, 1)
        return one_hot

    def forward(self, batch_H, text, is_train=True, batch_max_length=120):
        """
        input:
            batch_H : contextual_feature H = hidden state of encoder. [batch_size x num_steps x num_classes]
            text : the text-index of each image. [batch_size x (max_length+2)]. +2 for [START], [END] token. text[:, 0] = [GO].
        output: probability distribution at each step [batch_size x num_steps x num_classes]
        """
        batch_size = batch_H.size(0)
        num_steps = batch_max_length + 1  # +1 for [START], [END] tokens

        output_hiddens = torch.FloatTensor(batch_size, num_steps, self.hidden_size).fill_(0).to(self.device)
        hidden = (torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(self.device),
                  torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(self.device))

        if is_train:
            text = torch.cat([torch.zeros(batch_size, 1).long().to(self.device), text], dim=1)
            for i in range(num_steps):
                # one-hot vectors for a i-th char. in a batch
                char_onehots = self._char_to_onehot(text[:, i], onehot_dim=self.num_classes)
                # hidden : decoder's hidden s_{t-1}, batch_H : encoder's hidden H, char_onehots : one-hot(y_{t-1})
                hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)
                output_hiddens[:, i, :] = hidden[0]  # LSTM hidden index (0: hidden, 1: cell)
            probs = self.generator(output_hiddens)

            return F.softmax(probs, dim=2)  # batch_size x num_steps x num_classes
        else:
            ''''''
            k = self.cfg.beam_search_k
            targets = torch.LongTensor(batch_size).fill_(0).to(self.device)  # [START] token
            branches = [BeamSearchBranch(i) for i in range(batch_size * k)]

            ''' index 0 '''
            char_onehots = self._char_to_onehot(targets, onehot_dim=self.num_classes)
            hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)
            probs_step = F.softmax(self.generator(hidden[0]), dim=-1)
            values, indices = torch.topk(probs_step, k, dim=-1)
            indices = torch.flatten(indices, 0, 1)
            values = torch.flatten(values, 0, 1)

            ''' update '''
            hidden = [hidden[0], hidden[1]]
            hidden[0] = torch.repeat_interleave(hidden[0], repeats=k, dim=0)
            hidden[1] = torch.repeat_interleave(hidden[1], repeats=k, dim=0)
            batch_H = torch.repeat_interleave(batch_H, repeats=k, dim=0)
            targets = copy(indices)

            for i, b in enumerate(branches):
                b.insert(indices[i], values[i])
                # b.updateHidden(hidden, i)

            # 1 : [END]
            for i in range(1, num_steps):
                char_onehots = self._char_to_onehot(targets, onehot_dim=self.num_classes)
                hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)
                probs_step = F.softmax(self.generator(hidden[0]), dim=-1)

                h1, h2 = [], []
                for j in range(0, batch_size*k, k):
                    values, indices = torch.topk(probs_step[j:j+k,:], k, dim=-1)
                    values = torch.flatten(values, 0, 1)
                    indices = torch.flatten(indices, 0, 1)

                    temp = []
                    for t in range(k*k):
                        # tup = (t, score)
                        tup = (t, branches[j + t//k].getScore()*values[t])
                        temp.append(tup)
                    temp.sort(key=lambda tup: tup[1], reverse=True)  # ascending

                    new_branches = []
                    for t, tup in enumerate(temp[:k]):
                        new_branches.append(branches[j + tup[0]//k].copy())
                        if indices[tup[0]].item() != 1:  # <END> Token
                            new_branches[t].insert(indices[tup[0]], values[tup[0]])
                        # branches[j+t].updateHidden(hidden, j + tup[0]//k)
                        targets[j+t] = indices[tup[0]]
                        h1.append(hidden[0][j + tup[0]//k])
                        h2.append(hidden[1][j + tup[0]//k])
                    branches[j:j+k] = new_branches
                hidden = [torch.stack(h1), torch.stack(h2)]

            # return the bests
            ret = []
            for i in range(0, batch_size*k, k):
                winner = sorted(branches[i:i+k], key=lambda b: b.getScore())[-1]
                ret.append(winner)
            return ret
            # targets = torch.LongTensor(batch_size).fill_(0).to(self.device)  # [START] token
            # probs = torch.FloatTensor(batch_size, num_steps, self.num_classes).fill_(0).to(self.device)

            # for i in range(num_steps):
                # char_onehots = self._char_to_onehot(targets, onehot_dim=self.num_classes)
                # hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)
                # probs_step = self.generator(hidden[0])
                # probs[:, i, :] = probs_step
                # next_input = probs_step.argmax(1)
                # targets = next_input


class AttentionCell(nn.Module):

    def __init__(self, input_size, hidden_size, num_embeddings):
        super(AttentionCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)  # either i2i or h2h should have bias
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.LSTMCell(input_size + num_embeddings, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_onehots):
        # [batch_size x num_encoder_step x num_channel] -> [batch_size x num_encoder_step x hidden_size]
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = self.h2h(prev_hidden[0]).unsqueeze(1)
        e = self.score(torch.tanh(batch_H_proj + prev_hidden_proj))  # batch_size x num_encoder_step * 1

        alpha = F.softmax(e, dim=1)
        context = torch.bmm(alpha.permute(0, 2, 1), batch_H).squeeze(1)  # batch_size x num_channel
        concat_context = torch.cat([context, char_onehots], 1)  # batch_size x (num_channel + num_embedding)

        '''Decoder RNN'''
        cur_hidden = self.rnn(concat_context, prev_hidden)
        return cur_hidden, alpha


class BeamSearchBranch(object):
    def __init__(self, batch_id):
        self.id = batch_id
        self.sentence = []
        self.wait = None
        self.score = 1.
        self.hidden = None

    def __len__(self):
        return len(self.sentence)

    def __repr__(self):
        return  ' '.join(list(map(str, map(int, self.sentence)))) \
                + ' | {:.3}'.format(float(self.score))

    def insert(self, index, score):
        self.sentence.append(deepcopy(index))
        self.score *= score

    def getScore(self, alpha=0.6, min_length=3):
        p = (1+len(self.sentence))**alpha / (1+min_length)**alpha
        return self.score * p

    def updateHidden(self, hidden, idx):
        self.hidden = [hidden[0][idx], hidden[1][idx]]

    def copy(self):
        return deepcopy(self)

    def getSentence(self):
        return self.sentence

