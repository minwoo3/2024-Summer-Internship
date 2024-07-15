import cv2
import torch
import sys, os
import argparse
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torchvision.utils
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from data.RoadStatusModelDS_v2 import RoadStatusDataset

clean_train_ratio, clean_val_ratio = 0.7, 0.15
dirty_train_ratio, dirty_val_ratio = 0.5, 0.5
train_batch_sz, val_batch_sz, test_batch_sz = 32, 2, 20
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
    front_path = '/media/ampere_2_1/Seojeonghyun/T7/2024-Summer-Internship'
else:
    front_path = '/media/rideflux/T7/2024-Summer-Internship'
    
# dirty_ds = DirtyRoadDataset(front_path + '/2024-Summer-Internship/CameraBlockDL/data/dirty.csv')
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
test_dl = DataLoader(dataset = dirty_test_ds, batch_size = test_batch_sz, shuffle= True, drop_last = False)

classes = ['clean', 'dirty']

model = torch.load('resultmodel.pt')
model.eval()
cols, rows = 8, 4
fig = plt.figure(figsize = (rows, cols))
for i in range(1, cols*rows + 1):
    data_index = np.random.randint(len(dirty_test_ds))
    input_img = dirty_test_ds[data_index][0].unsqueeze(dim=0).to("cuda")
    output = model(input_img)
    # print(output.shape)
    _, argmax = torch.max(output, 1)
    pred = classes[argmax.item()]
    label = classes[dirty_test_ds[data_index][1]]

    fig.add_subplot(rows, cols, i)
    plt.title(f'{label}/{pred}')
    # if pred == label:
    #     plt.title(f'Gt: {label}, pred: {pred}')
    # else:
    #     plt.title(f'{pred} / {label}')
    # print(test_ds[data_index][0].permute(1,2,0).shape)
    # plot_img = cv2.cvtColor(np.array(test_ds[data_index][0].permute(1,2,0)), cv2.COLOR_BGR2RGB)
    plot_img = test_ds[data_index][0].permute(1,2,0)
    plt.imshow(plot_img)
    plt.axis('off')

plt.show()