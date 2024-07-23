import os
import cv2
import csv, getpass
import numpy as np
import PIL.Image as pil
from torch.utils.data import Dataset
from torchvision import transforms

class RoadStatusDataset(Dataset):
    def __init__(self, annotation_file, transform_flag = ''):

        with open(annotation_file,'r',newline='') as f:
            data = list(csv.reader(f))
        self.img_path, self.img_label = [], []
        for path, label in data:
            self.img_path.append(path)
            self.img_label.append(int(label))
        self.username = getpass.getuser()
        self.t7_dir = f'/media/{self.username}/T7/2024-Summer-Internship'
        self.sata_dir = f'/media/{self.username}/sata-ssd'
        self.transform_flag = transform_flag
        
        if self.transform_flag == 'ptf':
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
            ])
        elif self.transform_flag == 'crop':
            self.transform = transforms.Compose([
                transforms.Resize((720,1280)),
                transforms.Lambda(lambda img: img.crop((0, int(img.height*0.5), img.width, int(img.height*0.9)))),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((720,1280)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
            ])

    def perspectiveTF(self,img):
        img = np.array(img)
        img = cv2.resize(img,(1280,720))
        pts1 = np.float32([[330, 500], [950, 500], [100, 650], [1200, 650]])
        tl, tr, bl, br = pts1[0], pts1[1], pts1[2], pts1[3]
        w1 = abs(br[0]-bl[0])
        w2 = abs(tr[0]-tl[0])
        width = int(max([w1,w2]))
        h1 = abs(br[1]-tr[1])
        h2 = abs(bl[1]-tl[1])
        height = int(max([h1,h2]))
        pts2 = np.float32([[0,0],[width-1,0],[0, height-1],[width-1,height-1]])
        transform_mat = cv2.getPerspectiveTransform(pts1,pts2)
        result = cv2.warpPerspective(np.array(img), transform_mat, (width, height))
        return pil.fromarray(result)
    
    def __getitem__(self, idx):
        if ('NIA' in self.img_path[idx]) or ('벚꽃' in self.img_path[idx]):
            img = pil.open(self.t7_dir+self.img_path[idx])
        elif ('GeneralCase' in self.img_path[idx]):
            img = pil.open(self.sata_dir+self.img_path[idx])

        if self.transform_flag == 'ptf':
            img = self.perspectiveTF(img)
        x = self.transform(img)
        y = self.img_label[idx]
        return x, y, self.img_path[idx]
    
    def __len__(self):
        return len(self.img_path)

