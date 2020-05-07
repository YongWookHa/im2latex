"""
BTCKRW Dataset, DataLoader implementation
"""
import torch
import numpy as np
import pandas as pd
import logging
import os
from utils.utils import format_dataset
from datetime import datetime
from torch.utils.data import Dataset, DataLoader


class Im2LatexDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def __getitem__(self, index):
        return self.x[index]
    
    def finalize(self):
        pass
