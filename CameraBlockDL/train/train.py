import torch
import os
import sys
import lightning as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

import os

from CameraBlockDL.model.pl_module import BlockageClassificationModule as block_module
from CameraBlockDL.model.pl_module import get_module
from CameraBlockDL.data.dataset import DataModule
from lightning.pytorch.callbacks import ModelCheckpoint
from CameraBlockDL.miscs.pformat import pprint

# monitor_criteria = ['val_loss/dataloader_idx_1', 'val_loss/dataloader_idx_0']

# devel = True
devel = False
# float32 행렬 곱셈의 내부 정밀도를 반환
torch.set_float32_matmul_precision('medium')
def train(cfg):
    # torch.manual_seed : 난수생성을 위한 시드 설정
    torch.manual_seed(cfg.seed)

    train_dir = f"{cfg.train.dirpath}/{cfg.id}"

    if not os.path.isdir(train_dir):
        os.makedirs(train_dir)

    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.train.monitor,
        dirpath=train_dir,
        filename=cfg.train.filename
    )

    min_epochs = cfg.train.min_epochs
    max_epochs = cfg.train.max_epochs
    # Trainer(): Automatically enabling/disabling grads / Running the training, validation and test dataloaders
    # Calling the Callbacks at the appropriate times / Putting batches and computations on the correct devices
    trainer = pl.Trainer(callbacks=[EarlyStopping(monitor=cfg.train.monitor, patience=cfg.train.patience), checkpoint_callback],
                         fast_dev_run=devel,
                         min_epochs=min_epochs,
                         max_epochs=max_epochs,
                         accelerator='gpu',
                         num_sanity_val_steps=0)


    dm = DataModule(cfg)
    try:
        model = get_module(cfg)
        print("Model loaded from checkpoint")
    except:
        model = block_module(cfg) # cfg.labels.loss_weights 로 하는게 좋은지?
        print("Can't load model, creating new one")

    trainer.fit(model, dm)
    ckpt_path = cfg.test.checkpoint_path

    # if running test time at the same time.
    if devel:
        # "/home/test1804/PycharmProjects/st_intern_hyunkunkim/TrafficlightClassificationDL/model/archived/00000065/mobilenet_v3_small-tl_model-epoch=00-val_loss=0.00.ckpt"
        # "/home/test1804/PycharmProjects/st_intern_hyunkunkim/TrafficlightClassificationDL/model/archived/00000046/mobilenet_v2-tl_model-epoch=02-val_loss=0.00.ckpt"
        # "/home/test1804/PycharmProjects/st_intern_hyunkunkim/TrafficlightClassificationDL/model/archived//00000080/efficientnet_b0-tl_model-epoch=00-val_loss=0.00.ckpt"

        # ckpt_path = cfg.train.dirpath + "/00000065/mobilenet_v3_small-tl_model-epoch=00-val_loss=0.00.ckpt"
        # ckpt_path = cfg.train.dirpath + "/00000046/mobilenet_v2-tl_model-epoch=02-val_loss=0.00.ckpt"
        ckpt_path = cfg.train.dirpath + "/00000080/efficientnet_b0-tl_model-epoch=00-val_loss=0.00.ckpt"
        result = trainer.test(ckpt_path=ckpt_path, datamodule=dm)
    else:
        if ckpt_path == "best":
            cfg_ckp_dir = f"{cfg.train.dirpath}/{cfg.id}"
            ckpt_path = f"{cfg_ckp_dir}/{os.listdir(cfg_ckp_dir)[-1]}"
            cfg.test.checkpoint_path = ckpt_path
        # result = trainer.test(ckpt_path="best", datamodule=dm)
        result = trainer.test(ckpt_path=ckpt_path, datamodule=dm)
    pprint(result, ["OKBLUE"])

    del trainer
    del model
    del dm
