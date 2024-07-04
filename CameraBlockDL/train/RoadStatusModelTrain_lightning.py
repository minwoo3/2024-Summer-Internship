import torch
import sys, os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.RoadStatusModelNN_lightning import ResnetModule, CNNModule
from data.RoadStatusModelDS_lightning import RoadStadusDataModule

# module = ResnetModule(opt=1e-5)
opt=1e-5
module = CNNModule(opt)
module_name = module.__class__.__name__
batch_size = 16
datamodule = RoadStadusDataModule(batch_size)

# checkpoint_callback = ModelCheckpoint(
#     monitor='val/loss',
#     dirpath='checkpoints/',
#     filename='best-checkpoint',
#     save_top_k=1,
#     mode='min'
# )

# early_stopping_callback = EarlyStopping(
#     monitor='val/loss',
#     patience=5,
#     verbose=True,
#     mode='min'
# )

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

trainer.save_checkpoint(f'{module_name}_epochs_{max_epochs}_lr_{opt}_crop.ckpt')

######## Test #######
# best_model_path = checkpoint_callback.best_model_path
# module = ResnetModule.load_from_checkpoint(best_model_path)

# trainer.test(module, datamodule)
