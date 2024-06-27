import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torch.nn.functional as F
import lightning as pl
from easydict import EasyDict as edict
import numpy as np
# from CameraBlockDL.configs.config import save_cfg
from tqdm import tqdm
import csv
import PIL.Image as pil
import argparse
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm

class DirtyRoadDataset(Dataset):
    def __init__(self, annotation_file):
        with open(annotation_file,'r',newline='') as f:
            self.data = list(csv.reader(f))
        self.data_path = np.array(self.data).T[0]
        self.label = list(map(int,np.array(self.data).T[1]))
        self.transform = transforms.Compose([
            transforms.Resize((180, 320)),
            transforms.ToTensor()
        ])
    def __getitem__(self, idx):
        im = pil.open(self.data_path[idx]).convert("RGB")
        im = im.resize((180, 320))
        x = self.transform(im)
        y = self.label[idx]
        return x, y
    
    def __len__(self):
        return len(self.data)

class CleanRoadDataset(Dataset):
    def __init__(self, annotation_file):
        with open(annotation_file,'r',newline='') as f:
            self.data = list(csv.reader(f))
        self.data_path = np.array(self.data).T[0]
        self.label = list(map(int,np.array(self.data).T[1]))
        self.transform = transforms.Compose([
            transforms.Resize((180, 320)),
            transforms.ToTensor()
        ])
    def __getitem__(self, idx):
        im = pil.open(self.data_path[idx]).convert("RGB")
        im = im.resize((180, 320))
        x = self.transform(im)
        y = self.label[idx]
        return x, y
    
    def __len__(self):
        return len(self.data)

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Linear(6400, 2)
    
    def forward(self, x):
        x = self.sequential(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
############ ResNet50 ###############
def conv_block(in_dim, out_dim, kernel_size, activation, stride=1):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride),
        nn.BatchNorm2d(out_dim),
        activation,
    )
    return model

class BottleNeck(nn.Module):
    def __init__(self,in_dim,mid_dim,out_dim,activation,down=False):
        super(BottleNeck,self).__init__()
        self.down=down
        # 피처맵의 크기가 감소하는 경우
        if self.down:
            self.layer = nn.Sequential(
              conv_block(in_dim,mid_dim,1,activation,stride=2),
              conv_block(mid_dim,mid_dim,3,activation,stride=1),
              conv_block(mid_dim,out_dim,1,activation,stride=1),
            )
            
            # 피처맵 크기 + 채널을 맞춰주는 부분
            self.downsample = nn.Conv2d(in_dim,out_dim,kernel_size=1,stride=2)
            
        # 피처맵의 크기가 그대로인 경우
        else:
            self.layer = nn.Sequential(
                conv_block(in_dim,mid_dim,1,activation,stride=1),
                conv_block(mid_dim,mid_dim,3,activation,stride=1),
                conv_block(mid_dim,out_dim,1,activation,stride=1),
            )
        self.dim_equalizer = nn.Conv2d(in_dim, out_dim, kernel_size=1)
    
    def forward(self,x):
        if self.down:
            downsample = self.downsample(x)
            out = self.layer(x)
            out = out + downsample
        else:
            out = self.layer(x)
            if x.size() is not out.size():
                x = self.dim_equalizer(x)
            out = out + x
        return out

class ResNet50(nn.Module):
    def __init__(self, base_dim, num_classes=10):
        super(ResNet50, self).__init__()
        self.activation = nn.ReLU()
        self.layer_1 = nn.Sequential(
            nn.Conv2d(3,base_dim,7,2,3),
            nn.ReLU(),
            nn.MaxPool2d(3,2,1),
        )
        self.layer_2 = nn.Sequential(
            BottleNeck(base_dim,base_dim,base_dim*4,self.activation),
            BottleNeck(base_dim*4,base_dim,base_dim*4,self.activation),
            BottleNeck(base_dim*4,base_dim,base_dim*4,self.activation,down=True),
        )   
        self.layer_3 = nn.Sequential(
            BottleNeck(base_dim*4,base_dim*2,base_dim*8,self.activation),
            BottleNeck(base_dim*8,base_dim*2,base_dim*8,self.activation),
            BottleNeck(base_dim*8,base_dim*2,base_dim*8,self.activation),
            BottleNeck(base_dim*8,base_dim*2,base_dim*8,self.activation,down=True),
        )
        self.layer_4 = nn.Sequential(
            BottleNeck(base_dim*8,base_dim*4,base_dim*16,self.activation),
            BottleNeck(base_dim*16,base_dim*4,base_dim*16,self.activation),
            BottleNeck(base_dim*16,base_dim*4,base_dim*16,self.activation),            
            BottleNeck(base_dim*16,base_dim*4,base_dim*16,self.activation),
            BottleNeck(base_dim*16,base_dim*4,base_dim*16,self.activation),
            BottleNeck(base_dim*16,base_dim*4,base_dim*16,self.activation,down=True),
        )
        self.layer_5 = nn.Sequential(
            BottleNeck(base_dim*16,base_dim*8,base_dim*32,self.activation),
            BottleNeck(base_dim*32,base_dim*8,base_dim*32,self.activation),
            BottleNeck(base_dim*32,base_dim*8,base_dim*32,self.activation),
        )
        self.avgpool = nn.AvgPool2d(1,1) 
        self.fc_layer = nn.Linear(base_dim*32,num_classes)
        
    def forward(self, x):
        out = self.layer_1(x)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = self.layer_5(out)
        out = self.avgpool(out)
        out = out.view(train_batch_sz,-1)
        out = self.fc_layer(out)
        return out


if __name__ == "__main__" :
    ###### Parameters ######
    clean_train_ratio, clean_val_ratio = 0.7, 0.15
    dirty_train_ratio, dirty_val_ratio = 0.5, 0.5
    train_batch_sz, val_batch_sz, test_batch_sz = 32, 2, 2
    epochs, learning_rate = 10, 1e-5

    ###### Arguments ######
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', '--train', dest='train', action = 'store_true')
    parser.add_argument('-val', '--validation', dest='validation', action = 'store_true')
    parser.add_argument('-test', '--test', dest='test', action = 'store_true')
    args = parser.parse_args()

    ########## 총 45851개 ###########
    dirty_ds = DirtyRoadDataset('/home/rideflux/2024-Summer-Internship/CameraBlockDL/data/dirty.csv')
    clean_ds = CleanRoadDataset('/home/rideflux/2024-Summer-Internship/CameraBlockDL/data/clean.csv')

    dirty_ds_sz, clean_ds_sz= len(dirty_ds), len(clean_ds)

    clean_train_sz, clean_val_sz = int(clean_ds_sz * clean_train_ratio), int(clean_ds_sz * clean_val_ratio)
    clean_test_sz = clean_ds_sz - clean_train_sz - clean_val_sz

    dirty_train_sz = int(dirty_ds_sz * dirty_train_ratio)
    dirty_val_sz = dirty_ds_sz - dirty_train_sz

    clean_train_ds, clean_val_ds, clean_test_ds = random_split(clean_ds, [clean_train_sz, clean_val_sz, clean_test_sz])
    dirty_train_ds, dirty_val_ds = random_split(dirty_ds, [dirty_train_sz, dirty_val_sz])
    dirty_test_ds = dirty_val_ds
    
    ########## 총 48797개 ################
    ########## train / val / test ########
    # clean : [27972, 5994, 5994]
    # dirty : [2945, 2946, 2946]
    train_ds = dirty_train_ds + clean_train_ds
    val_ds = dirty_val_ds + clean_val_ds
    test_ds = dirty_test_ds + clean_test_ds

    train_dl = DataLoader(dataset = train_ds, batch_size = train_batch_sz, shuffle= True, drop_last = False)
    val_dl = DataLoader(dataset = val_ds, batch_size = val_batch_sz, shuffle= True, drop_last = False)
    test_ds = DataLoader(dataset = test_ds, batch_size = test_batch_sz, shuffle= True, drop_last = False)

    # 1배치 시각화
    # images, labels = next(iter(train_dl))
    # figure = plt.figure(figsize=(12,8))
    # cols, rows = 8, 4
    # label_map = ["clean", "dirty"]
    # for i in range(1, cols*rows + 1):
    #     sample_idx = torch.randint(len(images), size = (1,)).item()
    #     img, label = images[sample_idx], labels[sample_idx].item()
    #     if label == 0.0: label = 0 
    #     else: label = 1
    #     figure.add_subplot(rows, cols, i)
    #     plt.title(label_map[int(label)])
    #     plt.axis("off")
    #     plt.imshow(torch.permute(img, (1,2,0)))
    # plt.show()


    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained = True)
    model = CNNModel()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    
    if args.train:
        for epoch in range(epochs):
            model.train()
            running_loss, corr = 0, 0
            progress_bar = tqdm(train_dl, desc = f'Epoch {epoch +1}/{epochs}')
            for img, label in progress_bar:
                img, label = img.to(device), label.to(device)
                optimizer.zero_grad()
                output = model(img)
                loss = loss_fn(output, label.long())
                loss.backward()
                optimizer.step()
                _ , pred = output.max(dim=1)
                corr += pred.eq(label).sum().item()
                running_loss += loss.item()*img.size(0)
            acc = corr/len(train_dl.dataset)
            print(f'Epoch {epoch+1}/{epochs}, Training Accuracy: {acc:.4f}, loss: {running_loss / len(train_dl.dataset):.4f}')

        PATH = '/home/rideflux/2024-Summer-Internship/CameraBlockDL'
        torch.save(model,PATH + 'model.pt')   
        # for epoch in range(epochs+1):
        #     for batch_idx, samples in enumerate(train_dl):
        #         x_train, y_train = samples
                
        #         prediction = model(x_train)
        #         print(prediction)
        #         cost = F.cross_entropy(prediction, y_train)

        #         optimizer.zero_grad()
        #         cost.backward()
        #         optimizer.step()

        #         print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(epoch, epochs, batch_idx+1, len(train_dl),cost.item()))

