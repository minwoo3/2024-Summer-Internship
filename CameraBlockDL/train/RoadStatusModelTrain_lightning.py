import time
import argparse
import torch
import sys, os
import getpass
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping
from torch.utils.data import Dataset, DataLoader, random_split
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from pytorch_lightning.loggers import WandbLogger
from model.RoadStatusModelNN_lightning import ResnetModule
from data.RoadStatusModelDS_lightning import RoadStadusDataModule
import wandb

wandb.init()

module = ResnetModule(opt=1e-5)
datamodule = RoadStadusDataModule(batch_size=32)

checkpoint_callback = ModelCheckpoint(
    monitor='val/loss',
    dirpath='checkpoints/',
    filename='best-checkpoint',
    save_top_k=1,
    mode='min'
)

early_stopping_callback = EarlyStopping(
    monitor='val/loss',
    patience=10,
    verbose=True,
    mode='min'
)

wandb_logger = WandbLogger(project = "RoadStatusModel")

trainer = pl.Trainer(
    accelerator="gpu",
    devices=[0],
    max_epochs=100,
    logger=wandb_logger,
    callbacks=[checkpoint_callback, early_stopping_callback]
)

######## Train & Validation #######
trainer.fit(module, datamodule)

######## Test #######
best_model_path = checkpoint_callback.best_model_path
module = ResnetModule.load_from_checkpoint(best_model_path)

trainer.test(module, datamodule)