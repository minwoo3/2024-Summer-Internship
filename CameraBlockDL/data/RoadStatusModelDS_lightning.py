import os
import sys
import getpass
import numpy as np
import PIL.Image as pil
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from data.RoadStatusModelDS_v2 import RoadStatusDataset

class RoadStadusDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super(RoadStadusDataModule, self).__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        front_path = f'/media/{getpass.getuser()}/T7/2024-Summer-Internship'
        self.clean_train_ds = RoadStatusDataset(front_path + '/scene/clean_train.csv')
        self.clean_val_ds = RoadStatusDataset(front_path + '/scene/clean_val.csv')
        self.clean_test_ds = RoadStatusDataset(front_path + '/scene/clean_test.csv')
        self.dirty_train_ds = RoadStatusDataset(front_path + '/scene/dirty_train.csv')
        self.dirty_val_ds = RoadStatusDataset(front_path + '/scene/dirty_val.csv')
        self.dirty_test_ds = RoadStatusDataset(front_path + '/scene/dirty_test.csv')

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = self.clean_train_ds + self.dirty_train_ds
            self.valid_dataset = self.clean_val_ds + self.dirty_val_ds

        if stage == 'test' or stage is None:
            self.test_dataset = self.clean_test_ds + self.dirty_test_ds
        
    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
    
    def val_dataloader(self):
        return DataLoader(dataset=self.valid_dataset,batch_size=self.batch_size,shuffle=False,drop_last=False)
    
    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset,batch_size=self.batch_size,shuffle=False,drop_last=False)
