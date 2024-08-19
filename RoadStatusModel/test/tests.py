import torch
torch.set_float32_matmul_precision('medium')
import os, sys
import getpass
import argparse
import pytorch_lightning as pl
from pytorch_lightning import Trainer
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.module import CNNModule, ResnetModule
from data.datamodule import RoadStatusDataModule

username = getpass.getuser()
ssd_ckpt_dir = f'/media/{username}/T7/2024-Summer-Internship/checkpoint/cnn'
opt, batch_size, transform_flag = 1e-5, 16, 'mask'

ckpt_list = ['CNNModule_008','CNNModule_009','CNNModule_010','CNNModule_011','CNNModule_013','CNNModule_013_2','CNNModule_014','CNNModule_014_2','CNNModule_015','CNNModule_016']

for ckpt in ckpt_list:
    datamodule = RoadStatusDataModule(ckpt_name = ckpt, batch_size = batch_size, transform_flag = transform_flag)
    datamodule.setup(stage='fit')
    example_img, _, _ = datamodule.train_dataset[0]
    transformed_img_size = example_img.shape[-2:]  # (height, width)
    module = CNNModule.load_from_checkpoint(f'{ssd_ckpt_dir}/{ckpt}.ckpt', img_width=transformed_img_size[1], img_height=transformed_img_size[0], opt=opt, ckpt_name = ckpt)
    trainer = Trainer(accelerator='gpu', devices=1)
    trainer.test(module, dataloaders=datamodule)

