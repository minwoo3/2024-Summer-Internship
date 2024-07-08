import csv
import torch
import os, sys
import argparse
import getpass
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms.functional import to_pil_image
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

# 기존 모듈 임포트
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.RoadStatusModelNN_lightning import CNNModule
from data.RoadStatusModelDS_lightning import RoadStadusDataModule
from data.RoadStatusModelAT_v2 import annotate
from test.writer import csvwriter, txtwriter

username = getpass.getuser()

parser = argparse.ArgumentParser()
parser.add_argument('-update', '--update', dest='update', action='store_true')
args = parser.parse_args()

if args.update == True:
    nia_img_dir = f'/media/{username}/T7/2024-Summer-Internship/NIA2021'
    cbtree_img_dir = f'/media/{username}/T7/2024-Summer-Internship/벚꽃'
    clean_csv_save_dir = f'/media/{username}/T7/2024-Summer-Internship/scene/clean'
    dirty_csv_save_dir = f'/media/{username}/T7/2024-Summer-Internship/scene/dirty'
    annotate(nia_img_dir, cbtree_img_dir, clean_csv_save_dir, dirty_csv_save_dir)

# 모델 로드
model = CNNModule.load_from_checkpoint('path_to_your_checkpoint.ckpt')
batch_size = 16
datamodule = RoadStadusDataModule(batch_size)

# 테스트 실행
trainer = Trainer(gpus=1)
trainer.test(model, test_dataloaders=datamodule)
