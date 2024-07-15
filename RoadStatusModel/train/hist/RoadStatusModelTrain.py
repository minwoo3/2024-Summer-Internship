import time
import argparse
import torch
import sys, os
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from RoadStatusModel.model.model_v3 import CNNModel, conv_block, BottleNeck, ResNet50
from RoadStatusModel.data.hist.RoadStatusModelDS import DirtyRoadDataset, CleanRoadDataset
from data.RoadStatusModelDS_v2 import RoadStatusDataset

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
    parser.add_argument('-path', '--path', dest='path', action = 'store')
    args = parser.parse_args()

    ########## 총 45851개 ###########
    # 데이터 셋 불러오기
    if args.path == 'server':
        front_path = '/media/ampere_2_1/T7/2024-Summer-Internship'
    else:
        front_path = '/media/rideflux/T7/2024-Summer-Internship'
        
    clean_train_ds = RoadStatusDataset(front_path + '/scene/clean_train.csv')
    clean_val_ds = RoadStatusDataset(front_path + '/scene/clean_val.csv')
    clean_test_ds = RoadStatusDataset(front_path + '/scene/clean_test.csv')

    dirty_train_ds = RoadStatusDataset(front_path + '/scene/dirty_train.csv')
    dirty_val_ds = RoadStatusDataset(front_path + '/scene/dirty_val.csv')
    dirty_test_ds = RoadStatusDataset(front_path + '/scene/dirty_test.csv')
    
    print(f'Train / Valid / Test')
    print(f'{len(clean_train_ds)} / {len(clean_val_ds)} / {len(clean_test_ds)}')
    print(f'{len(dirty_train_ds)} / {len(dirty_val_ds)} / {len(dirty_test_ds)}')

    # Total dataset 만들기 = dirty + clean
    train_ds = dirty_train_ds + clean_train_ds
    val_ds = dirty_val_ds + clean_val_ds
    test_ds = dirty_test_ds + clean_test_ds

    # DataLoader 선언
    train_dl = DataLoader(dataset = train_ds, batch_size = train_batch_sz, shuffle= True, drop_last = False)
    val_dl = DataLoader(dataset = val_ds, batch_size = val_batch_sz, shuffle= True, drop_last = False)
    test_ds = DataLoader(dataset = test_ds, batch_size = test_batch_sz, shuffle= True, drop_last = False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CNNModel()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    start_time = time.time()
    running_loss_log = []
    if args.train:
        for epoch in range(epochs):
            model.train()
            running_loss, corr = 0, 0
            progress_bar = tqdm(train_dl, desc = f'Epoch {epoch +1}/{epochs}')
            for img, label, path in progress_bar:
                img, label = img.to(device), label.to(device)
                optimizer.zero_grad()
                output = model(img)
                loss = loss_fn(output, label.long())
                loss.backward()
                optimizer.step()
                _ , pred = output.max(dim=1)
                corr += pred.eq(label).sum().item()
                running_loss += loss.item()*img.size(0)
                running_loss_log.append(running_loss)
            acc = corr/len(train_dl.dataset)
            print(f'Epoch {epoch+1}/{epochs}, Training Accuracy: {acc:.4f}, loss: {running_loss / len(train_dl.dataset):.4f}')
        end_time = time.time()
        duration = end_time - start_time
        result_save_dir = front_path + '/CameraBlockDL/train'
        torch.save(model, result_save_dir + f'{time.time()}_model.pt')