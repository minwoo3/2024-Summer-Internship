import os
import sys
import getpass
import numpy as np
import PIL.Image as pil
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from data.RoadStatusModelDS_v2_30000 import RoadStatusDataset

class RoadStadusDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int =8, crop = False):
        super(RoadStadusDataModule, self).__init__()
        self.batch_size = batch_size
        self.front_path = f'{os.path.dirname(os.getcwd())}/data/30000.csv'
        self.crop = crop
    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = RoadStatusDataset(self.front_path)
            self.valid_dataset = RoadStatusDataset(self.front_path)
        if stage == 'test' or stage is None:         
            self.test_dataset = RoadStatusDataset(self.front_path)


    # num_worker: 
    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False, num_workers=16)
    
    def val_dataloader(self):
        return DataLoader(dataset=self.valid_dataset,batch_size=self.batch_size,shuffle=False,drop_last=False, num_workers=16)
    
    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset,batch_size=self.batch_size,shuffle=False,drop_last=False, num_workers=16)
