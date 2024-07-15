import torch
import sys, os
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.RoadStatusModelNN_lightning import MainModule

# data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # 1채널 이미지를 3채널로 변환
])
dataset = MNIST('', train=True, download=True, transform=transform)
mnist_train, mnist_val = random_split(dataset, [55000, 5000])
train_loader = DataLoader(mnist_train, batch_size=32)
val_loader = DataLoader(mnist_val, batch_size=32)

wandb_logger = WandbLogger(project="pl-mnist")

model = MainModule(opt = 1e-5)
trainer = pl.Trainer(
    accelerator="gpu",
    devices=[0],
    max_epochs=5,
    logger=wandb_logger
)
trainer.fit(model, train_loader)