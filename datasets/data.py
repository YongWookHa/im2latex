import torch
import numpy as np
import os

from tqdm import tqdm
from PIL import Image

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class Im2LatexDataset(Dataset):
    def __init__(self, config, mode="train"):
        super().__init__()
        self.config = config

        if mode == "train":
            formula_path = config.train_formula_path
            image_path = config.train_img_path
        elif mode == "valid":
            formula_path = config.valid_formula_path
            image_path = config.valid_img_path
        else:
            raise NotImplementedError

        # tokens
        self.NullTokenID, self.StartTokenID = 0, 1

        # load image and formula
        self.dataset = []
        self.id2token = {0:'[START]', 1:'[END]'}
        self.token2id = {'[START]':0, '[END]':1}
        cnt = 2
        print('Reading from {}'.format(formula_path))
        tqdm_bar = tqdm(enumerate(open(formula_path, 'r')), desc="DataLoading")
        for i, form in tqdm_bar:
            if config.debug and i > 500: break
            img_fn = "{}/{}.png".format(image_path, i)
            form = form.split()[:config.max_len]
            self.dataset.append((img_fn, form))

            # build vocab
            if mode == "train":
                for token in form:
                    if token not in self.token2id:
                        self.token2id[token] = cnt
                        self.id2token[cnt] = token
                        cnt += 1
        if mode == "train" and config.build_new_vocab:
            print('{} Words Vocab Built'.format(len(self.token2id)))
            torch.save({
                'id2token' : self.id2token,
                'token2id' : self.token2id
            }, os.path.join('experiments', config.exp_name, 'vocab.pkl'))
        else:
            try:
                vocab = torch.load(os.path.join('experiments',
                                                config.exp_name, 'vocab.pkl'))
                self.id2token = vocab['id2token']
                self.token2id = vocab['token2id']
            except FileNotFoundError:
                print("vocab file not found!")
                print("use built vocab..")

        # set vocab_size (call by reference)
        config.vocab_size = len(self.id2token)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    def finalize(self):
        pass

class custom_collate(object):
    def __init__(self, token2id, max_len):
        self.token2id = token2id
        self.max_len = max_len
        self.transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize([0.5], [0.5])])

    def __call__(self, batch):
        # filter the pictures that have different weight or height
        img_fn, formulas = zip(*batch)
        imgs = [self.transform(Image.open(fn).convert('L')) for fn in img_fn]

        # targets for calculating loss , end with END_TOKEN
        tgt = self.formulas2tensor(formulas, self.token2id)

        imgs = torch.stack(imgs, dim=0)
        return (imgs, tgt)

    def formulas2tensor(self, formulas, token2id):
        """convert formula to tensor"""

        batch_size = len(formulas)
        BOS, EOS = 0, 1
        tensors = torch.ones((batch_size, self.max_len+1), dtype=torch.long) * EOS
        for i, formula in enumerate(formulas):
            assert len(formula) <= self.max_len
            tensors[i][:len(formula)] = torch.tensor([token2id[token] for token in formula])
        return tensors

    # def add_start_token(self, formulas):
    #     return [['\\bos']+formula for formula in formulas]

    # def add_end_token(self, formulas):
    #     return [formula+['\\eos'] for formula in formulas]
