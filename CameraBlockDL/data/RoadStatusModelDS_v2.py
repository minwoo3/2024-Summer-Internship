from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import csv
import PIL.Image as pil
import os

class RoadStatusDataset(Dataset):
    def __init__(self, annotation_file):
        with open(annotation_file,'r',newline='') as f:
            data = list(csv.reader(f))
        for i in range(len(data)):
            data[i] = ''.join(data[i])
        self.img_path = data
        if 'clean' in annotation_file:
            self.label = [0 for _ in range(len(self.img_path))]
        elif 'dirty' in annotation_file:
            self.label = [1 for _ in range(len(self.img_path))]

        self.transform = transforms.Compose([
            # transforms.Resize((180, 320)),
            transforms.Resize((720, 1280)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
        ])
    def __getitem__(self, idx):
        im = pil.open(self.img_path[idx])
        x = self.transform(im)
        y = self.label[idx]
        return x, y, self.img_path[idx]
    
    def __len__(self):
        return len(self.img_path)

