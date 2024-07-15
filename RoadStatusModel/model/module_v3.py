import csv
import torch
import sys, os, io
import torchmetrics
import numpy as np
import torch.nn as nn
import PIL.Image as Image
from torchvision import transforms
import torch.optim as optim
from typing import Any, Optional
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchvision.models import resnet18, resnet34, resnet50

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from RoadStatusModel.model.model_v3 import CNNModel

def csvwriter(csv_dir, target_list):
    with open(csv_dir, 'w', newline="") as file:
        writer = csv.writer(file)
        writer.writerows(target_list)
    print(f'List Saved at {csv_dir} Succesfully')

def txtwriter(txt_dir, target_list):
    with open(txt_dir, 'w', newline="") as file:
        file.write('\n'.join(target_list))
    print(f'List Saved at {txt_dir} Succesfully')

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


class CNNModule(pl.LightningModule):
    def __init__(self, opt, img_width, img_height):
        super(CNNModule, self).__init__()
        self.opt = opt
        self.model = CNNModel(img_width, img_height)
        # self.featuremap = self.model.featuremap

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        im, label, path = batch 
        im, label = im.to(self.device), label.to(self.device)
        pred = self.model(im)
        train_loss = F.cross_entropy(pred, label)
        self.log('train/loss', train_loss, on_step=True, on_epoch=True, prog_bar=True)
        return train_loss 
    
    def validation_step(self, batch, batch_idx):
        im, label, path = batch
        im, label = im.to(self.device), label.to(self.device)
        with torch.no_grad():
            pred = self.model(im)
            val_loss = F.cross_entropy(pred, label)
        self.log('val/loss', val_loss, on_step=True, on_epoch=True, prog_bar=True)
        return val_loss
    
    def test_step(self, batch, batch_idx):
        ims, labels, paths = batch
        ims, labels = ims.to(self.device), labels.to(self.device)
        with torch.no_grad():
            output = self.model(ims)
            _, preds = torch.max(output, 1)
            tp, tn, fp, fn = 0, 0, 0, 0
            false_batch = []
            for i in range(len(labels)):
                pred = preds[i].item()
                label = labels[i].item()
                path = paths[i]

                if label == 0:
                    if pred == 0:
                        tp += 1
                    else:
                        fn += 1
                        false_batch.append(path)
                else:
                    if pred == 1:
                        tn += 1
                    else:
                        fp += 1
                        false_batch.append(path)

        return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'false_batch': false_batch}
    
    def test_epoch_end(self, outputs):
        tp = sum([output['tp'] for output in outputs])
        tn = sum([output['tn'] for output in outputs])
        fp = sum([output['fp'] for output in outputs])
        fn = sum([output['fn'] for output in outputs])
        false_batch = [path for output in outputs for path in output['false_batch']]

        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        recall = tp / (tp + fn)
        specificity = tn / (fp + tn)
        precision = tp / (tp + fp)
        f1 = 2 * precision * recall / (precision + recall)
        
        print(f'Test Finished\nTrue Positive: {tp}, True Negative: {tn}, False Positive: {fp}, False Negative: {fn}')
        print(f'Accuracy: {accuracy*100}%, Recall: {recall*100}%, Specificity: {specificity*100}%, Precision: {precision*100}%, F1: {f1}')
        
        txtwriter('result.txt', false_batch)

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
