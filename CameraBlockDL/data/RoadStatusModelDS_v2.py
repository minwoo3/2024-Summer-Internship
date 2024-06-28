from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import csv
import PIL.Image as pil


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
        im = pil.open(self.data_path[idx])
        im = im.resize((180, 320))
        x = self.transform(im)
        y = self.label[idx]
        return x, y, self.data_path[idx]
    
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
        im = pil.open(self.data_path[idx])
        im = im.resize((180, 320))
        x = self.transform(im)
        y = self.label[idx]
        return x, y, self.data_path[idx]
    
    def __len__(self):
        return len(self.data)
