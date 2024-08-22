import csv, torch, getpass, sys, os, io, torchmetrics
import numpy as np
import torch.nn as nn
import PIL.Image as Image
from torchvision import transforms
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import resnet18
from torchmetrics.classification import BinaryAccuracy
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.model import CNNModel

def csvwriter(csv_dir, target_list):
    with open(csv_dir, 'a', newline="") as file:
        writer = csv.writer(file)
        writer.writerows(target_list)
    print(f'List Saved at {csv_dir} Successfully')
    
class CNNModule(pl.LightningModule):
    def __init__(self, opt, img_width, img_height, ckpt_name):
        super(CNNModule, self).__init__()
        self.opt = opt
        self.model = CNNModel(img_width, img_height)
        self.confusion_matrix = torchmetrics.ConfusionMatrix(task = 'binary', num_classes=2)
        self.accuracy = BinaryAccuracy()
        
        self.class_amount = [110928,30588] # clean, dirty
        self.class_weight = [1- x/sum(self.class_amount) for x in self.class_amount]
        
        self.ckpt_name = ckpt_name
        self.ssd_dir = f'/media/{getpass.getuser()}/T7/2024-Summer-Internship/scene/{self.ckpt_name}'

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        im, label, path = batch 
        im, label = im.to(self.device), label.to(self.device)
        
        pred = self.model(im).squeeze(1)
        self.pos_weight = torch.tensor([self.class_amount[0]/self.class_amount[1]],device = 'cuda')
        train_loss = F.binary_cross_entropy_with_logits(pred, label.float(), pos_weight=self.pos_weight)
        
        batch_size = im.size(0)
        self.log('train/loss', train_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist = True)
        return train_loss 
    
    def validation_step(self, batch, batch_idx):
        im, label, path = batch 
        im, label = im.to(self.device), label.to(self.device)
        
        pred = self.model(im).squeeze(1)
        self.pos_weight = torch.tensor([self.class_amount[0]/self.class_amount[1]],device = 'cuda')
        val_loss = F.binary_cross_entropy_with_logits(pred, label.float(), pos_weight=self.pos_weight)
        
        batch_size = im.size(0)
        self.log('val/loss', val_loss, on_step=True, on_epoch=True, prog_bar=True,batch_size=batch_size, sync_dist = True)
        return val_loss
    
    def test_step(self, batch, batch_idx):
        ims, labels, paths = batch
        ims, labels = ims.to(self.device), labels.to(self.device)
        
        pred = self.model(ims).squeeze(1)
        
        pred_class = (torch.sigmoid(pred) > 0.5).long()
        self.confusion_matrix(pred_class, labels)
        self.accuracy(pred_class, labels)
        
        false_batch = []
        for i in range(len(labels)):
            pred = pred_class[i].item()
            label = labels[i].item()
            path = paths[i]
            if (label == 0 and pred == 1) or (label == 1 and pred == 0):
                false_batch.append([path,label])

        return false_batch
    
    def test_epoch_end(self, outputs):
        cm = self.confusion_matrix.compute().cpu()
        accuracy = self.accuracy.compute().cpu()
        print("Confusion Matrix:\n", cm.numpy())
        print("Test Accuracy:", accuracy.numpy())
        self.confusion_matrix.reset()
        self.accuracy.reset()
        false_batch = [path for batch in outputs for path in batch]
        csvwriter(f'{self.ssd_dir}/result.csv', false_batch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opt)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/loss'
            }
        }
    @property
    def featuremap(self):
        return self.model.featuremap

class ResnetModule(pl.LightningModule):
    def __init__(self, opt):
        super(ResnetModule, self).__init__()
        self.opt = opt
        self.model = resnet18()
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-2])
        self.confusion_matrix = torchmetrics.ConfusionMatrix(task = 'binary', num_classes=2)
        self._featuremap = None
    
    def forward(self, x):
        self._featuremap = self.feature_extractor(x)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        self.model.train()
        im, label, path = batch 
        pred = self.model(im)
        train_loss = F.cross_entropy(pred, label)
        self.log('train/loss', train_loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        return train_loss 
    
    def validation_step(self, batch, batch_idx):
        self.model.eval()
        im, label, path = batch
        pred = self.model(im)
        val_loss = F.cross_entropy(pred, label)
        self.log('val/loss', val_loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        return val_loss
    
    def test_step(self, batch, batch_idx):
        self.model.eval()
        im, label, path = batch
        pred = self.model(im)
        test_loss = F.cross_entropy(pred, label)
        self.log('test/loss', test_loss, on_step=True, on_epoch=True, prog_bar=True)
        pred_class = torch.argmax(pred, dim=1)
        
        self.confusion_matrix(pred_class, label)
        return test_loss
    
    def test_epoch_end(self, outputs):
        cm = self.confusion_matrix.compute().cpu()
        print("Confusion Matrix:\n", cm.numpy())
        
        # Log confusion matrix as an image
        fig, ax = plt.subplots()
        cax = ax.matshow(cm.numpy(), cmap='Blues')
        fig.colorbar(cax)
        for (i, j), val in np.ndenumerate(cm.numpy()):
            ax.text(j, i, f'{val:.0f}', ha='center', va='center')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        # Convert the plot to a tensor
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        image = Image.open(buf)
        image = transforms.ToTensor()(image)
        
        self.logger.experiment.add_image('Confusion matrix', image)
        
        self.confusion_matrix.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opt)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/loss'
            }
        }

    @property
    def featuremap(self):
        return self._featuremap

    @featuremap.setter
    def featuremap(self, value):
        self._featuremap = value


