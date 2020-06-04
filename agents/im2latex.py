import os
import torch
import numpy as np
import shutil
from functools import partial
from tqdm import tqdm
from torch import nn
from torch.backends import cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.nn.parallel import DataParallel

from tensorboardX import SummaryWriter

from utils.misc import print_cuda_statistics, get_device
from agents.base import BaseAgent

from datasets.data import Im2LatexDataset, custom_collate
from graphs.models.model import Im2LatexModel
from utils.utils import cal_loss

cudnn.benchmark = True


class Im2latex(BaseAgent):

    def __init__(self, cfg):
        super().__init__(cfg)
        # print_cuda_statistics()
        self.device = get_device()
        self.cfg = cfg

        # dataset
        train_dataset = Im2LatexDataset(cfg, mode="train")
        valid_dataset = Im2LatexDataset(cfg, mode="valid")
        self.id2token = valid_dataset.id2token
        self.token2id = valid_dataset.token2id

        collate = custom_collate(self.token2id, cfg.max_len)

        self.train_loader = DataLoader(train_dataset, batch_size=cfg.bs,
                            shuffle=cfg.data_shuffle, num_workers=cfg.num_w,
                            collate_fn=collate)
        self.valid_loader = DataLoader(valid_dataset, batch_size=cfg.bs,
                            shuffle=cfg.data_shuffle, num_workers=cfg.num_w,
                            collate_fn=collate)

        # define models
        self.model = Im2LatexModel(cfg, len(self.id2token))  # fill the parameters
        self.model = DataParallel(self.model)
        # define criterion
        self.criterion = cal_loss

        # define optimizers for both generator and discriminator
        self.optimizer = torch.optim.Adam(  params=self.model.parameters(),
                                            lr=cfg.lr,
                                            betas=( cfg.adam_beta_1,
                                                    cfg.adam_beta_2))

        # initialize counter
        self.current_epoch = 1
        self.current_iteration = 1
        self.best_loss = 100
        self.best_info = ''

        # set the manual seed for torch
        torch.cuda.manual_seed_all(self.cfg.seed)
        if self.cfg.cuda:
            self.model = self.model.cuda().to(self.device)
            self.logger.info("Program will run on *****GPU-CUDA***** ")
        else:
            self.logger.info("Program will run on *****CPU*****\n")

        # Model Loading from cfg if not found start from scratch.
        self.exp_dir = os.path.join('./experiments', cfg.exp_name)
        self.load_checkpoint(cfg.checkpoint_filename)
        # Summary Writer
        self.summary_writer = SummaryWriter(log_dir=os.path.join(self.exp_dir,
                                                              'summaries'))

    def load_checkpoint(self, file_name):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        try:
            self.logger.info("Loading checkpoint '{}'".format(file_name))
            checkpoint = torch.load(file_name, map_location=self.device)

            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.model.load_state_dict(checkpoint['model'], strict=False)
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            info = "Checkpoint loaded successfully from "
            self.logger.info(info + "'{}' at (epoch {}) at (iteration {})\n"
              .format(file_name, checkpoint['epoch'], checkpoint['iteration']))

        except OSError as e:
            self.logger.info("Checkpoint not found in '{}'.".format(file_name))
            self.logger.info("**First time to train**")

    def save_checkpoint(self, file_name="checkpoint.pth", is_best=False):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current
                        checkpoint's accuracy is the best so far
        :return:
        """
        state = {
            'epoch': self.current_epoch,
            'iteration': self.current_iteration,
            'model' : self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

        # save the state
        checkpoint_dir = os.path.join(self.exp_dir, 'checkpoints')
        file_name = "e{}-i{}.pt".format(self.current_epoch,
                                        self.current_iteration)
        torch.save(state, os.path.join(checkpoint_dir, file_name))

        if is_best:
            shutil.copyfile(os.path.join(checkpoint_dir, file_name),
                            os.path.join(checkpoint_dir, 'best.pt'))

    def run(self):
        """
        The main operator
        :return:
        """
        try:
            if self.cfg.mode == 'train':
                self.train()
            elif self.cfg.mode == 'valid':
                self.predict()

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        """
        Main training loop
        :return:
        """
        for e in range(self.current_epoch, self.cfg.epochs+1):
            self.train_one_epoch()
            self.validate()
            self.current_epoch += 1

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        tqdm_bar = tqdm(enumerate(self.train_loader),
                        total=len(self.train_loader))

        self.model.train()
        avg_loss = 0
        for i, (imgs, tgt4training, tgt4cal_loss) in tqdm_bar:
            imgs = imgs.to(self.device).float()
            tgt4training = tgt4training.to(self.device).long()
            tgt4cal_loss = tgt4cal_loss.to(self.device).long()

            logits = self.model(imgs, tgt4training, is_train=True) # [B, MAXLEN, VOCABSIZE]
            loss = self.criterion(logits, tgt4cal_loss)

            # L2 regularization
            reg_loss = None
            for param in self.model.parameters():
                if reg_loss is None:
                    reg_loss = 0.5 * torch.sum(param**2)
                else:
                    reg_loss = reg_loss + 0.5 * param.norm(2)**2
            loss += reg_loss * 0.9
            avg_loss += loss.item()
            loss.backward()
            self.optimizer.step()

            # save if best
            if self.current_epoch % self.cfg.log_freq == 0:
                avg_loss = avg_loss / self.cfg.log_freq
                tqdm_bar.set_description("loss: {}".format(avg_loss))
                if  avg_loss < self.best_loss:
                    self.save_checkpoint()
                    self.best_loss = avg_loss
                    self.best_info = 'best: e{}_i{}'.format(self.current_epoch,
                                                        self.current_iteration)
                avg_loss = 0

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        tqdm_bar = tqdm(enumerate(self.valid_loader),
                        total=len(self.valid_loader))
        self.model.eval()
        with torch.no_grad():
            for i, (imgs, tgt4training, tgt4cal_loss) in tqdm_bar:
                imgs = imgs.to(self.device).float()

                tgt4cal_loss = tgt4cal_loss.to(self.device).long()

                logits = self.model(imgs) # [B, MAXLEN, VOCABSIZE]

                loss = self.criterion(tgt4cal_loss, logits)
                reg_loss = None
                for param in model.parameters():
                    if reg_loss is None:
                        reg_loss = 0.5 * torch.sum(param**2)
                    else:
                        reg_loss = reg_loss + 0.5 * param.norm(2)**2
                loss += reg_loss * 0.9
                tqdm_bar.set_description('loss={}'.format(loss))
                self.optimizer.step()

    def predict(self):
        """
        get predict results
        :return:
        """
        pass

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process,
        the operator and the data loader
        :return:
        """
        print(self.best_info)
        pass
