import torch
import sys, os
import argparse
import pytorch_lightning as pl
from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.module_v3 import ResnetModule, CNNModule
from data.datamodule_v3 import RoadStadusDataModule

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', dest='model', action = 'store')
parser.add_argument('-t', '--transform', dest = 'transform', action = 'store', default = '')
args = parser.parse_args()
opt, batch_size = 1e-5, 16
datamodule = RoadStadusDataModule(batch_size = batch_size, transform_flag = args.transform)

datamodule.setup(stage='fit')

# Get transformed image size
example_img, _, _ = datamodule.train_dataset[0]
transformed_img_size = example_img.shape[-2:]  # (height, width)
print(transformed_img_size)
if args.model in ['cnn','CNN']:
    module = CNNModule(img_width=transformed_img_size[1], img_height=transformed_img_size[0], opt=opt)
elif args.model in ['resnet','res','ResNet']:
    module = ResnetModule(opt)
else:
    raise ValueError("Invalid model name. Choose from ['cnn', 'CNN', 'resnet', 'res', 'ResNet']")
    
module_name = module.__class__.__name__

torch.cuda.empty_cache()

accelerator="gpu"
gpus = 2
strategy='ddp'
max_epochs=20
callbacks=[TQDMProgressBar()]
precision=16

trainer = pl.Trainer(
    accelerator=accelerator,
    gpus=gpus,
    strategy=strategy,
    max_epochs=max_epochs,
    callbacks=callbacks,
    precision=precision
)

######## Train & Validation #######
trainer.fit(module, datamodule)

now = datetime.now()

trainer.save_checkpoint(f'{module_name}_{now.date()}_epochs_{max_epochs}.ckpt')
