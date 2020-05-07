import os
import torch
import numpy as np
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader, random_split

from tensorboardX import SummaryWriter

from utils.misc import print_cuda_statistics, get_device
from agents.base import BaseAgent

from datasets.data import Im2LatexDataset
from graphs.model import Im2LatexModel
from utils import utils

cudnn.benchmark = True


class IM2LATEX(BaseAgent):

    def __init__(self, cfg):
        super().__init__(cfg)
        print_cuda_statistics()
        self.device = get_device()

        # define models
        self.model = Im2LatexModel()  # fill the parameters

        # define data_loader 1 or 2
        # 1
        # tr_dataset = custom_dataset(cfg.tr_data_pth)
        # te_dataset = custom_dataset(cfg.te_data_pth)
        
        # 2
        dataset = Im2LatexDataset(cfg.data_pth)
        tr_dataset, te_dataset = random_split(dataset, 
                                               [train_size, test_size])

        self.tr_loader = DataLoader(tr_dataset, batch_size=cfg.bs, 
                              shuffle=cfg.data_shuffle, num_workers=cfg.num_w)
        self.te_loader = DataLoader(te_dataset, batch_size=cfg.bs, 
                              shuffle=cfg.data_shuffle, num_workers=cfg.num_w)

        # define criterion
        self.criterion = torch.nn.NLLLoss()

        # define optimizers for both generator and discriminator
        self.optimizer = torch.optim.Adam(  lr=opt.lr, 
                                            betas=( cfg.adam_beta_1,
                                                    cfg.adam_beta_2))

        # initialize counter
        self.current_epoch = 1
        self.current_iteration = 1
        self.best_metric = 0  # loss or accuracy or etc
        self.best_info = ''

        # set the manual seed for torch
        torch.cuda.manual_seed_all(self.cfg.seed)
        if self.cuda:
            self.model = self.model.cuda()
            self.logger.info("Program will run on *****GPU-CUDA***** ")
        else:
            self.logger.info("Program will run on *****CPU*****\n")

        # Model Loading from cfg if not found start from scratch.
        self.exp_dir = os.path.join('./experiments', cfg.exp_name)
        self.load_checkpoint(self.cfg.checkpoint_file)
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

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=False):
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
            self.train()

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        """
        Main training loop
        :return:
        """
        for e in range(self.current_epoch, self.cfg.epochs+1):
            train_one_epoch()
            validate()
            self.current_epoch += 1

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """

        loss = self.criterion(gt, predict)
        reg_loss = None
        for param in model.parameters():
            if reg_loss is None:
                reg_loss = 0.5 * torch.sum(param**2)
            else:
                reg_loss = reg_loss + 0.5 * param.norm(2)**2
        loss += reg_loss * 0.9

        pass

    def validate(self):
        """
        One cycle of model validation
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
