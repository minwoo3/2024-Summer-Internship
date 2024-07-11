import torch
import sys, os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.RoadStatusModelNN_lightning import ResnetModule, CNNModule
from CameraBlockDL.data.RoadStatusModelDM_lightning_30000 import RoadStadusDataModule

# module = ResnetModule(opt=1e-5)
opt=1e-5
module = CNNModule(opt,1280,720)
# module = ResnetModule(opt)


module = CNNModule.load_from_checkpoint('CNNModule_epochs_20_lr_1e-05.ckpt', opt)
module_name = module.__class__.__name__
batch_size = 16
datamodule = RoadStadusDataModule(batch_size)

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

trainer.save_checkpoint(f'{module_name}_epochs_{max_epochs}_lr_{opt}_30000.ckpt')

######## Test #######
# best_model_path = checkpoint_callback.best_model_path
# module = ResnetModule.load_from_checkpoint(best_model_path)

# trainer.test(module, datamodule)
