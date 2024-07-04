import csv
import torch
import sys, os
import argparse
import getpass
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
import torchvision.utils
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from data.RoadStatusModelDS_v2 import RoadStatusDataset
from torchvision.transforms.functional import to_pil_image
from data.RoadStatusModelAT_v2 import annotate

def csvwriter(csv_dir, target_list):
    with open(csv_dir, 'w', newline="") as file:
        writer = csv.writer(file)
        writer.writerows(target_list)
    print(f'List Saved at {csv_dir} Succesfully')

def txtwriter(txt_dir, target_list):
    with open(txt_dir, 'w', newline="") as file:
        file.write('\n'.join(target_list))
    print(f'List Saved at {txt_dir} Succesfully')


username = getpass.getuser()

parser = argparse.ArgumentParser()
parser.add_argument('-update', '--update', dest='update', action = 'store_true')
args = parser.parse_args()

if args.update == True:
    nia_img_dir = f'/media/{username}/T7/2024-Summer-Internship/NIA2021'
    cbtree_img_dir = f'/media/{username}/T7/2024-Summer-Internship/벚꽃'
    clean_csv_save_dir = f'/media/{username}/T7/2024-Summer-Internship/scene/clean'
    dirty_csv_save_dir = f'/media/{username}/T7/2024-Summer-Internship/scene/dirty'
    annotate(nia_img_dir, cbtree_img_dir, clean_csv_save_dir, dirty_csv_save_dir)

train_batch_sz, val_batch_sz, test_batch_sz = 32, 2, 20
epochs, learning_rate = 10, 1e-5

# curr_folder = os.getcwd()
# parent_folder = os.path.dirname(curr_folder)

#### SSD Dir ####
front_path = f'/media/{username}/T7/2024-Summer-Internship'
clean_test_ds = RoadStatusDataset(front_path + '/scene/clean_test.csv')
dirty_test_ds = RoadStatusDataset(front_path + '/scene/dirty_test.csv')

test_ds = dirty_test_ds + clean_test_ds

test_dl = DataLoader(dataset = test_ds, batch_size = test_batch_sz, shuffle= True, drop_last = False)

classes = ['clean', 'dirty']

model = torch.load('1719907458.2549872_model.pt')

model.eval()
false_batch = []
tp, tn, fp, fn = 0, 0, 0, 0

for imgs, labels, paths in tqdm(test_dl):
    fn, fp = False, False
    test_batch = []
    for i in range(test_batch_sz):
        try:
            img, label, path = imgs[i], labels[i], paths[i]
            output = model(img.unsqueeze(dim=0).to("cuda"))
            _, argmax = torch.max(output, 1)
            pred = argmax.item()
            test_batch.append(path)
            if label == 0:
                if pred == 0: 
                    tp += 1
                else: 
                    fn += 1
                    fn = True
            else:
                if pred == 1:
                    tn += 1
                else:
                    fp += 1
                    fp = True

        except IndexError:
            pass

    if fn == True or fp == True:
        false_batch.append(test_batch)

print('Test Finished')
print(f'True Positive: {tp}, True Negative: {tn}, False Positive: {fp}, False Negative: {fn}')
accuracy = (tp+tn)/(tp+tn+fp+fn)
recall = tp/(tp+fn)
specificity = tn/(fp+tn)
precision = tp/(tp+fp)
f1 = 2*precision*recall/(precision+recall)
print(f'Accuracy : {accuracy*100}%, Recall : {recall*100}%, Specificity : {specificity*100}%, Precision = {precision*100}, f1 = {f1}')

csvwriter('result.csv', false_batch)
txtwriter('result.txt', false_batch)
