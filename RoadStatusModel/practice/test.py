import sys, os
import argparse
import getpass
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.module_v3 import CNNModule, ResnetModule
from data.datamodule_v3 import RoadStadusDataModule

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', dest='model', action = 'store')
args = parser.parse_args()
opt, batch_size = 1e-5, 16
datamodule = RoadStadusDataModule(batch_size)

datamodule.setup(stage='fit')

example_img, _, _ = datamodule.train_dataset[0]
transformed_img_size = example_img.shape[-2:]

if args.model in ['cnn','CNN']:
    module = CNNModule(img_width=transformed_img_size[1], img_height=transformed_img_size[0], opt=opt)
elif args.model in ['resnet','res','ResNet']:
    module = ResnetModule(opt)
else:
    raise ValueError("Invalid model name. Choose from ['cnn', 'CNN', 'resnet', 'res', 'ResNet']")
    
module_name = module.__class__.__name__