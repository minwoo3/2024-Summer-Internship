from typing import Any, Optional
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
import sys, os
from torchvision.models import resnet34, resnet50
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.RoadStatusModelNN import CNNModel
from test.writer import csvwriter, txtwriter
class ResnetModule(pl.LightningModule):
	def __init__(self, opt):
		super(ResnetModule, self).__init__()
		self.opt=opt 	
		self.model = resnet34()
		
	def forward(self, x):
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
		return test_loss
	
	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.model.parameters(), lr = self.opt)
		# 학습률 조정
		scheduler = ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.1, patience = 10, verbose = True)
		return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val/loss'
        }
	

class CNNModule(pl.LightningModule):
    def __init__(self, opt):
        super(CNNModule, self).__init__()
        self.opt = opt
        self.model = CNNModel()
        
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

                accuracy = (tp + tn) / (tp + tn + fp + fn)
                recall = tp / (tp + fn)
                specificity = tn / (fp + tn)
                precision = tp / (tp + fp)
                f1 = 2 * precision * recall / (precision + recall)
                print(f'Test Finished\nTrue Positive: {tp}, True Negative: {tn}, False Positive: {fp}, False Negative: {fn}')
                print(f'Accuracy: {accuracy*100}%, Recall: {recall*100}%, Specificity: {specificity*100}%, Precision: {precision*100}%, F1: {f1}')

            # csvwriter('result.csv', false_batch)
            txtwriter('result.txt', false_batch)
        return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'false_batch': false_batch}
    
    def test_epoch_end(self, outputs):
        pass

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