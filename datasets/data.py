"""
BTCKRW Dataset, DataLoader implementation
"""
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
        
        if config.mode == "train":
            formula_path = config.train_formula_path
            image_path = config.train_img_path
        elif config.mode == "valid":
            formula_path = config.valid_formula_path
            image_path = config.valid_img_path
        else:
            raise NotImplementedError

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), 
                                                            (0.5, 0.5, 0.5))])

        # tokens
        self.NullTokenID, self.StartTokenID = 0, 1

        # load image and formula
        self.dataset = []
        self.id2token = {0:'\\bos', 1:'\\eos'}
        self.token2id = {'\\bos':0, '\\eos':1}
        cnt = 2
        tqdm_bar = tqdm(enumerate(open(formula_path, 'r'), desc="DataLoading"))
        for i, form in tqdm_bar:
            img = Image.open("{}/{}.png".format(image_path, i))
            form = form.split()[:config.max_len]
            self.dataset.append((transform(img), form))

            # build vocab
            if config.mode == "train":
                for token in form:
                    if token not in self.token2id:
                        self.token2id[token] = cnt
                        self.id2token[cnt] = token
                        cnt += 1
        if config.mode == "train":
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
                raise


    def __getitem__(self, index):
        return self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)

    def finalize(self):
        pass

def collate_fn(token2id, batch):
    # filter the pictures that have different weight or height
    size = batch[0][0].size()
    batch = [img_formula for img_formula in batch
             if img_formula[0].size() == size]

    imgs, formulas = zip(*batch)
    formulas = [formula.split() for formula in formulas]
    # targets for training , begin with START_TOKEN
    tgt4training = formulas2tensor(add_start_token(formulas), token2id)
    # targets for calculating loss , end with END_TOKEN
    tgt4cal_loss = formulas2tensor(add_end_token(formulas), token2id)
    imgs = torch.stack(imgs, dim=0)
    return imgs, tgt4training, tgt4cal_loss


def formulas2tensor(formulas, token2id):
    """convert formula to tensor"""

    batch_size = len(formulas)
    max_len = len(formulas[0])
    BOS, EOS = 0, 1
    tensors = torch.ones(batch_size, max_len, dtype=torch.long) * EOS
    for i, formula in enumerate(formulas):
        for j, token in enumerate(formula):
            tensors[i][j] = token2id[token]
    return tensors


def add_start_token(formulas):
    return [['\\bos']+formula for formula in formulas]

def add_end_token(formulas):
    return [formula+['\\eos'] for formula in formulas]
