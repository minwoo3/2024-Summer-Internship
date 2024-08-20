import torch
torch.set_float32_matmul_precision('medium')
import os, sys
import getpass
import argparse
import pytorch_lightning as pl
from pytorch_lightning import Trainer
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.module import CNNModule, ResnetModule
from data.datamodule_v3 import RoadStatusDataModule

username = getpass.getuser()

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', dest='model', action = 'store')
parser.add_argument('-c', '--ckpt', dest='checkpoint', action = 'store')
parser.add_argument('-t', '--transform', dest='transform', action = 'store')

args = parser.parse_args()
opt, batch_size = 1e-5, 16
datamodule = RoadStatusDataModule(ckpt_name = args.checkpoint, batch_size = batch_size, transform_flag = args.transform)

datamodule.setup(stage='fit')

# Get transformed image size
example_img, _, _, _ = datamodule.train_dataset[0]
transformed_img_size = example_img.shape[-2:]  # (height, width)
# print(transformed_img_size)

if args.model in ['cnn','CNN']:
    ssd_dir = f'/media/{username}/T7/2024-Summer-Internship/checkpoint/cnn'
    module = CNNModule.load_from_checkpoint(f'{ssd_dir}/{args.checkpoint}.ckpt', 
                                            img_width=transformed_img_size[1], 
                                            img_height=transformed_img_size[0], opt=opt, ckpt_name = args.checkpoint)
elif args.model in ['resnet','res','ResNet']:
    ssd_dir = f'/media/{username}/T7/2024-Summer-Internship/checkpoint/resnet'
    module = ResnetModule.load_from_checkpoint(f'{ssd_dir}/{args.checkpoint}.ckpt',
                                                 opt = 1e-5, strict = False)
else:
    raise ValueError("Invalid model name. Choose from ['cnn', 'CNN', 'resnet', 'res', 'ResNet']")
    
module_name = module.__class__.__name__

trainer = Trainer(accelerator='gpu', devices=1)
trainer.test(module, dataloaders=datamodule)

################ Result ###################
# CNNModule_epochs_20_lr_1e-05_crop.ckpt: 
# True Positive: 6000, True Negative: 7601, False Positive: 0, False Negative: 0
# Accuracy: 100.0%, Recall: 100.0%, Specificity: 100.0%, Precision: 100.0%, F1: 1.0
