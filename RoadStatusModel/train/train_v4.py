import torch
torch.set_float32_matmul_precision('medium')
import sys, os
import argparse, getpass
import pytorch_lightning as pl
from datetime import datetime
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.module_v4 import ResnetModule, CNNModule
from data.datamodule_v3 import RoadStadusDataModule

username = getpass.getuser()
ssd_dir = f'/media/{username}/T7/2024-Summer-Internship'
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', dest='model', action = 'store')
parser.add_argument('-c', '--ckpt', dest='checkpoint', action = 'store')
parser.add_argument('-t', '--transform', dest = 'transform', action = 'store', default = '')
args = parser.parse_args()
opt, batch_size = 1e-5, 16

if args.model in ['cnn','CNN']:
    ckpt_dir = f'{ssd_dir}/checkpoint/cnn'
    if args.checkpoint in ['', None]: 
        num_ckpt = len(os.listdir(ckpt_dir))
        ckpt_name = '{0}_{1:03d}'.format('CNNModule',num_ckpt)
    else:
        ckpt_name = args.checkpoint
    print(ckpt_name,'will be save at', ckpt_dir+'/'+ckpt_name+'.ckpt')
    datamodule = RoadStadusDataModule(ckpt_name = ckpt_name, batch_size = batch_size, transform_flag = args.transform)
    datamodule.setup(stage='fit')
    example_img, _, _, _ = datamodule.train_dataset[0]
    transformed_img_size = example_img.shape[-2:]  # (height, width)

    module = CNNModule(img_width=transformed_img_size[1], img_height=transformed_img_size[0], opt=opt, ckpt_name = ckpt_name)
    
elif args.model in ['resnet','res','ResNet']:
    ckpt_dir = f'{ssd_dir}/checkpoint/resnet'
    if args.checkpoint in ['', None]:
        num_ckpt = len(os.listdir(ckpt_dir))
        ckpt_name = '{0}_{1:03d}'.format('ResnetModule',num_ckpt)
    else:
         ckpt_name = args.checkpoint
    print(ckpt_name,'will be save at', ckpt_dir+ckpt_name+'.ckpt')
    datamodule = RoadStadusDataModule(ckpt_name = ckpt_name, batch_size = batch_size, transform_flag = args.transform)
    datamodule.setup(stage='fit')
    example_img, _, _ = datamodule.train_dataset[0]
    transformed_img_size = example_img.shape[-2:]  # (height, width)

    module = ResnetModule(opt)

else:
    raise ValueError("Invalid model name. Choose from ['cnn', 'CNN', 'resnet', 'res', 'ResNet']")

torch.cuda.empty_cache()


# accelerator="gpu"
# gpus = 1
# # strategy='ddp'
# max_epochs=1

accelerator="gpu"
devices = 2
strategy=DDPStrategy(find_unused_parameters=False)
max_epochs=20
callbacks=[TQDMProgressBar()]
precision=16

trainer = pl.Trainer(
    accelerator=accelerator,
    devices=devices,
    strategy=strategy,
    max_epochs=max_epochs,
    callbacks=callbacks,
    precision=precision
)


######## Train & Validation #######
trainer.fit(module, datamodule)

# now = datetime.now()
trainer.save_checkpoint(f'{ckpt_dir}/{ckpt_name}.ckpt')
print(ckpt_name,'is saved at', ssd_dir)