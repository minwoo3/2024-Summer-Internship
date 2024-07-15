import os, sys
import getpass
import pytorch_lightning as pl
from pytorch_lightning import Trainer
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from RoadStatusModel.model.module_v3 import CNNModule, ResnetModule
from RoadStatusModel.data.datamodule_v3 import RoadStadusDataModule

username = getpass.getuser()

model = CNNModule.load_from_checkpoint('CNNModule_epochs_20_lr_1e-05_crop.ckpt', opt = 1e-5, img_width = 1280, img_height = 720)
# model = ResnetModule.load_from_checkpoint('ResnetModule_epochs_20_lr_1e-05.ckpt', opt = 1e-5, strict = False)
batch_size = 16
datamodule = RoadStadusDataModule(batch_size)

trainer = Trainer(gpus=1)
trainer.test(model, dataloaders=datamodule)

################ Result ###################
# CNNModule_epochs_20_lr_1e-05_crop.ckpt: 
# True Positive: 6000, True Negative: 7601, False Positive: 0, False Negative: 0
# Accuracy: 100.0%, Recall: 100.0%, Specificity: 100.0%, Precision: 100.0%, F1: 1.0
