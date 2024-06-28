import cv2
import torch
import sys, os
import argparse
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import PIL.Image as pil
import torchvision.utils
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from data.RoadStatusModelDS import DirtyRoadDataset, CleanRoadDataset


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
    front_path = '/home/ampere_2_1/Seojeonghyun'
else:
    front_path = '/home/rideflux'
    
dirty_ds = DirtyRoadDataset(front_path + '/2024-Summer-Internship/CameraBlockDL/data/dirty.csv')
clean_ds = CleanRoadDataset(front_path + '/2024-Summer-Internship/CameraBlockDL/data/clean.csv')

# 두 데이터 셋 비율 설정
dirty_ds_sz, clean_ds_sz= len(dirty_ds), len(clean_ds)

clean_train_sz, clean_val_sz = int(clean_ds_sz * clean_train_ratio), int(clean_ds_sz * clean_val_ratio)
clean_test_sz = clean_ds_sz - clean_train_sz - clean_val_sz

dirty_train_sz = int(dirty_ds_sz * dirty_train_ratio)
dirty_val_sz = dirty_ds_sz - dirty_train_sz

clean_train_ds, clean_val_ds, clean_test_ds = random_split(clean_ds, [clean_train_sz, clean_val_sz, clean_test_sz])
dirty_train_ds, dirty_val_ds = random_split(dirty_ds, [dirty_train_sz, dirty_val_sz])
dirty_test_ds = dirty_val_ds

# Total dataset 만들기 = dirty + clean
train_ds = dirty_train_ds + clean_train_ds
val_ds = dirty_val_ds + clean_val_ds
test_ds = dirty_test_ds + clean_test_ds

# DataLoader 선언
train_dl = DataLoader(dataset = train_ds, batch_size = train_batch_sz, shuffle= True, drop_last = False)
val_dl = DataLoader(dataset = val_ds, batch_size = val_batch_sz, shuffle= True, drop_last = False)
test_dl = DataLoader(dataset = test_ds, batch_size = test_batch_sz, shuffle= True, drop_last = False)
# dirty_test_dl = DataLoader(dataset = dirty_test_ds, batch_size = 1, shuffle= True, drop_last = False)
classes = ['clean', 'dirty']

model = torch.load('resultmodel.pt')
model.eval()
cols, rows = 8, 4
fn_list, fp_list = [], []
tp, tn, fp, fn = 0, 0, 0, 0

for imgs, labels, paths in tqdm(test_dl):
    for i in range(test_batch_sz):
        img, label, path = imgs[i], labels[i], paths[i]
        output = model(img.unsqueeze(dim=0).to("cuda"))
        _, argmax = torch.max(output, 1)
        pred = argmax.item()
        # clean:Positive, dirty: Negative
        if label == 0:
            if pred == 0:
                tp += 1
            else: 
                fn += 1
                fn_list.append(path)
        else:
            if pred == 1:
                tn += 1
            else:
                fp += 1
                fp_list.append(path)

print(f'True Positive: {tp}, True Negative: {tn}, False Positive: {fp}, False Negative: {fn}')
accuracy = (tp+tn)/(tp+tn+fp+fn)
recall = tp/(tp+fn)
specificity = tn/(fp+tn)
precision = tp/(tp+fp)
f1 = 2*precision*recall/(precision+recall)
print(f'Accuracy : {accuracy*100}%, Recall : {recall*100}%, Specificity : {specificity*100}%, Precision = {precision*100}, f1 = {f1}')

if fn_list:
    print(fn_list)

if fp_list:
    print(fp_list)
    
    # folder_path = '/home/rideflux/2024-Summer-Internship/CameraBlockDL/result/FalseNegative'
    # file_count = len(os.listdir(folder_path))
    # for path in fn_list:
    #     img = pil.open(path)
    #     img.save(f'fn_image_{file_count}.jpg')
# True Positive: 5987, True Negative: 2943, False Positive: 3, False Negative: 7
# Accuracy : 99.88814317673378%, Recall : 99.88321654988322%, Specificity : 99.89816700610999%, Precision = 99.94991652754591, f1 = 0.9991655540720961

# dirty_test_ds 로만 test
# True Positive: 0, False Positive
# True Negative:    False Negative 
# 
# 
# True Negative: 2941, False Positive: 5, False Negative: 0
# Accuracy : 0.9983027834351663%, Recall : -, Specificity : 99.89816700610999%, Precision = -, f1 = -