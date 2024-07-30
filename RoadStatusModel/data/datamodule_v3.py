import os
import sys
import getpass
import numpy as np
import PIL.Image as pil
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from data.dataset_v4 import RoadStatusDataset

class RoadStadusDataModule(pl.LightningDataModule):
    def __init__(self, ckpt_name, batch_size: int =8, transform_flag: str =''):
        super(RoadStadusDataModule, self).__init__()
        self.batch_size = batch_size
        self.username = getpass.getuser()
        self.front_path = f'/media/{self.username}/T7/2024-Summer-Internship/scene/{ckpt_name}'
        self.transform_flag = transform_flag
        
    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = RoadStatusDataset(self.front_path+'/train.csv',self.transform_flag)
            self.valid_dataset = RoadStatusDataset(self.front_path+'/val.csv',self.transform_flag)
        if stage == 'test' or stage is None:         
            self.test_dataset = RoadStatusDataset(self.front_path+'/test.csv',self.transform_flag)


    # num_worker: 
    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False, num_workers=16, pin_memory = True)
    
    def val_dataloader(self):
        return DataLoader(dataset=self.valid_dataset,batch_size=self.batch_size,shuffle=False,drop_last=False, num_workers=16, pin_memory = True)
    
    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset,batch_size=self.batch_size,shuffle=False,drop_last=False, num_workers=16, pin_memory = True)
