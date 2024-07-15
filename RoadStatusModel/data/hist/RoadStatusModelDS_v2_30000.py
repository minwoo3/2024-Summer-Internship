import os
import csv
import numpy as np
import PIL.Image as pil
from torch.utils.data import Dataset
from torchvision import transforms

class RoadStatusDataset(Dataset):
    def __init__(self, annotation_file):

        with open(annotation_file,'r',newline='') as f:
            data = list(csv.reader(f))
        self.img_path, self.img_label = [], []
        # 30000,dirty
        for path, label in data:
            scene_path = f'/home/ampere_2_1/GeneralCase/Raw/{path}/camera_0'
            # 1.jpg ...
            scene_img_list = os.listdir(scene_path)
            for scene_img in scene_img_list:
                # '/home/ampere21/GeneralCase/Raw/30000/camera_0/1.jpg'
                self.img_path.append(scene_path+"/"+scene_img)
                if label == 'dirty':
                    self.img_label.append(1)
                else:
                    self.img_label.append(0)

        self.transform = transforms.Compose([
            transforms.Resize((720, 1280)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
        ])

    def __getitem__(self, idx):
        im = pil.open(self.img_path[idx])
        x = self.transform(im)
        y = self.img_label[idx]
        return x, y, self.img_path[idx]
    
    def __len__(self):
        return len(self.img_path)

