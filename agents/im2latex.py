import os
import torch
import numpy as np
import shutil
from functools import partial
from tqdm import tqdm
from torch.backends import cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.nn.parallel import DataParallel

from tensorboardX import SummaryWriter

from utils.misc import get_device
from agents.base import BaseAgent

from datasets.data import Im2LatexDataset, custom_collate
from graphs.models.model import Im2LatexModel
from utils.utils import cal_loss

cudnn.benchmark = True


class Im2latex(BaseAgent):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.device = get_device()
        cfg.device = self.device
        self.cfg = cfg

        # dataset
        train_dataset = Im2LatexDataset(cfg, mode="train")
        self.id2token = train_dataset.id2token
        self.token2id = train_dataset.token2id

        collate = custom_collate(self.token2id, cfg.max_len)

        self.train_loader = DataLoader(train_dataset, batch_size=cfg.bs,
                            shuffle=cfg.data_shuffle, num_workers=cfg.num_w,
                            collate_fn=collate, drop_last=True)
        if cfg.valid_img_path != "":
            valid_dataset = Im2LatexDataset(cfg, mode="valid",
                                            vocab={'id2token': self.id2token,
                                                   'token2id': self.token2id})
            self.valid_loader = DataLoader(valid_dataset,
                                           batch_size=cfg.bs//cfg.beam_search_k,
                                           shuffle=cfg.data_shuffle,
                                           num_workers=cfg.num_w,
                                           collate_fn=collate,
                                           drop_last=True)

        # define models
        self.model = Im2LatexModel(cfg)  # fill the parameters
        # weight initialization setting
        for name, param in self.model.named_parameters():
            if 'localization_fc2' in name:
                print(f'Skip {name} as it is already initialized')
                continue
            try:
                if 'bias' in name:
                    torch.nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    torch.nn.init.kaiming_normal_(param)
            except Exception as e:  # for batchnorm.
                if 'weight' in name:
                    param.data.fill_(1)
                continue

        self.model = DataParallel(self.model)
        # define criterion
        self.criterion = cal_loss

        self.optimizer = torch.optim.Adam(  params=self.model.parameters(),
                                            lr=cfg.lr,
                                            betas=( cfg.adam_beta_1,
                                                    cfg.adam_beta_2))

        milestones = cfg.milestones
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                              milestones,
                                                              gamma=cfg.gamma,
                                                              verbose=True)

        # initialize counter
        self.current_epoch = 1
        self.current_iteration = 1
        self.best_metric = 100
        self.best_info = ''

        # set the manual seed for torch
        torch.cuda.manual_seed_all(self.cfg.seed)
        if self.cfg.cuda:
            self.model = self.model.to(self.device)
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
            'vocab' : self.id2token,
            'optimizer': self.optimizer.state_dict()
        }

        # save the state
        checkpoint_dir = os.path.join(self.exp_dir, 'checkpoints')
        if is_best:
            torch.save(state, os.path.join(checkpoint_dir, 'best.pt'))
            self.best_info = 'best: e{}_i{}'.format(self.current_epoch,
                                                    self.current_iteration)
        else:
            file_name = "e{}-i{}.pt".format(self.current_epoch,
                                            self.current_iteration)
            torch.save(state, os.path.join(checkpoint_dir, file_name))

    def run(self):
        """
        The main operator
        :return:
        """
        try:
            if self.cfg.mode == 'train':
                self.train()
            elif self.cfg.mode == 'predict':
                self.predict()

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        """
        Main training loop
        :return:
        """
        prev_perplexity = 0
        for e in range(self.current_epoch, self.cfg.epochs+1):
            this_perplexity = self.train_one_epoch()
            self.save_checkpoint()
            self.scheduler.step()
            self.current_epoch += 1

        if self.cfg.valid_img_path:
            self.validate()

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        tqdm_bar = tqdm(enumerate(self.train_loader, 1),
                        total=len(self.train_loader))

        self.model.train()
        last_avg_perplexity, avg_perplexity = 0, 0
        for i, (imgs, tgt) in tqdm_bar:
            imgs = imgs.float().to(self.device)
            tgt = tgt.long().to(self.device)

            # [B, MAXLEN, VOCABSIZE]
            logits = self.model(imgs, tgt, is_train=True)

            loss = self.criterion(logits, tgt)
            avg_perplexity += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()
            self.current_iteration += 1

            # logging
            if i % self.cfg.log_freq == 0:
                avg_perplexity = avg_perplexity / self.cfg.log_freq
                self.summary_writer.add_scalar('perplexity/train', avg_perplexity,
                                            global_step=self.current_iteration)
                self.summary_writer.add_scalar('lr', self.scheduler.get_last_lr(),
                                            global_step=self.current_iteration)
                tqdm_bar.set_description(
                    "e{} | avg_perplexity: {:.3f}".format(
                    self.current_epoch, avg_perplexity))

                # save if best
                if  avg_perplexity < self.best_metric:
                    self.save_checkpoint(is_best=True)

                    self.best_metric = avg_perplexity
                last_avg_perplexity = avg_perplexity
                avg_perplexity = 0

        mask = (tgt[0] != 2)
        pred = str(logits[0].argmax(1)[mask].cpu().detach().tolist())
        gt = str(tgt[0][mask].cpu().tolist())
        self.summary_writer.add_text('example/train', pred+'  \n'+gt,
                                     global_step=self.current_iteration)

        return last_avg_perplexity

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        tqdm_bar = tqdm(enumerate(self.valid_loader, 1),
                        total=len(self.valid_loader))
        self.model.eval()
        acc = 0
        with torch.no_grad():
            for i, (imgs, tgt) in tqdm_bar:
                imgs = imgs.to(self.device).float()
                tgt = tgt.to(self.device).long()

                logits = self.model(imgs, is_train=False).long() # [B, MAXLEN, VOCABSIZE]
                # mask = (tgt == 2)
                # tgt[mask] = 1
                # logits[mask] = 1
                acc += torch.all(tgt==logits, dim=1).sum() / imgs.size(0)
                # print('t', tgt)
                # print('l', logits)
                tqdm_bar.set_description('acc {:.4f}'.format(acc/i))
                if i % self.cfg.log_freq == 0:
                    self.summary_writer.add_scalar('accuracy/valid',
                                                   acc.item()/i,
                                                   global_step=self.current_iteration)

    def predict(self):
        """
        get predict results
        :return:
        """
        from torchvision import transforms
        from pathlib import Path
        from PIL import Image
        from time import time

        self.model.eval()
        transform = transforms.ToTensor()
        image_path = Path(self.cfg.test_img_path)

        t = time()
        with torch.no_grad():
            images = []
            imgPath = list(image_path.glob('*.jpg')) + list(image_path.glob('*.png'))
            for i, img in enumerate(imgPath):
                print(i, ':' ,img)
                img = Image.open(img)
                img = transform(img)
                images.append(img)
            images = torch.stack(images, dim=0)
            out = self.model(images)  # [B, max_len, vocab_size]
            out = out.argmax(2)

        for i, output in enumerate(out):
            print(i, ' '.join([self.id2token[out.item()] for out in output if out.item() != 1]))

        print(time()-t)

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process,
        the operator and the data loader
        :return:
        """
        print(self.best_info)
        pass
