import csv
import cv2
import torch
import math
import copy
import sys, os
import argparse
import getpass
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.module_v3 import CNNModule, ResnetModule
from data.datamodule_v3 import RoadStadusDataModule

def csvreader(csv_dir):
    with open(csv_dir, 'r', newline='') as f:
        data = list(csv.reader(f))
    paths, labels = [], []
    for path, label in data:
        paths.append(path)
        labels.append(int(label))
    print('Read CSV Successfully')
    return paths, labels

def ptf(pts,img):
    tl, tr, bl, br = pts
    pts1 = np.float32([tl, tr, bl, br])

    w1 = abs(br[0]-bl[0])
    w2 = abs(tr[0]-tl[0])
    width = int(max([w1,w2]))
    h1 = abs(br[1]-tr[1])
    h2 = abs(bl[1]-tl[1])
    height = int(max([h1,h2]))

    pts2 = np.float32([[0,0],[width-1,0],[0, height-1],[width-1,height-1]])

    transform_mat = cv2.getPerspectiveTransform(pts1,pts2)

    result = cv2.warpPerspective(img, transform_mat, (width, height))
    return result, transform_mat

class Viewer():
    def __init__(self, csv_path, index, model, transform_flag):
        self.csv_path, self.curr_i, self.transform_flag = csv_path, index, transform_flag
        self.t7_dir = f'/media/{username}/T7/2024-Summer-Internship'
        self.sata_dir = f'/media/{username}/sata-ssd'

        with open(csv_path,'r',newline='') as f:
            data = list(csv.reader(f))

        self.img_path, self.img_label= [], []
        for path, label in data:
            self.img_path.append(path)
            self.img_label.append(int(label))

        self.classes = ['clean','dirty']
        opt, batch_size = 1e-5, 16
        datamodule = RoadStadusDataModule(ckpt_name = args.checkpoint, batch_size = batch_size, 
                                                            transform_flag = self.transform_flag)
        datamodule.setup(stage='fit')

        example_img, _, _ = datamodule.train_dataset[0]
        self.img_height, self.img_width = example_img.shape[-2:]  # (height, width)

        if model in ['cnn','CNN']:
            ssd_dir = f'{self.t7_dir}/checkpoint/cnn'
            self.module = CNNModule.load_from_checkpoint(f'{ssd_dir}/{args.checkpoint}.ckpt',
                                                    img_width=self.img_width, img_height=self.img_height, 
                                                    opt=opt, ckpt_name = args.checkpoint)
        elif model in ['resnet','res','ResNet']:
            ssd_dir = f'{self.t7_dir}/checkpoint/resnet'
            self.module = ResnetModule.load_from_checkpoint(f'{ssd_dir}/{args.checkpoint}.ckpt',
                                                    opt = opt, strict = False)
        else:
            raise ValueError("Invalid model name. Choose from ['cnn', 'CNN', 'resnet', 'res', 'ResNet']")
    
        self.module_name = self.module.__class__.__name__
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
        ])

    def change_curr_dirs(self, dif):
        self.curr_i += dif
        if self.curr_i == len(self.img_path):
            print('end of list')
            self.curr_i -= dif
        elif self.curr_i == 0:
            print('first of list')
            self.curr_i += dif

    def drawCAM(self, width, height):
        activation_map = self.module.featuremap.squeeze().cpu()
        class_weights_gap = F.adaptive_avg_pool2d(activation_map,(1,1)).squeeze()
        cam = torch.zeros(activation_map.shape[1:], dtype=torch.float32)
        for i in range(len(class_weights_gap)):
            cam += class_weights_gap[i]*activation_map[i,:,:]

        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()
        cam = cam.detach().numpy()
        cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((width, height), Image.Resampling.LANCZOS)) / 255.0
        return cam_resized
        

    def view(self):
        while True:
            curr_img_path = self.img_path[self.curr_i]
            curr_img_label = self.img_label[self.curr_i]
            if 'NIA' in curr_img_path or '벚꽃' in curr_img_path:
                original_img = cv2.imread(self.t7_dir + curr_img_path)
            elif 'GeneralCase' in curr_img_path:
                original_img = cv2.imread(self.sata_dir + curr_img_path)
            
            original_img = cv2.resize(original_img,(1280, 720))
            img = original_img.copy()

            if self.transform_flag == 'ptf':
                pts1 = [[330, 500], [950, 500], [100, 650], [1200, 650]]
                img, transform_mat = ptf(pts1,img)
                inverse_mat = np.linalg.inv(transform_mat)

            elif self.transform_flag == 'crop':
                x_start, x_end = 0, img.shape[1]
                y_start, y_end = int(img.shape[0]*0.4), int(img.shape[0]*0.8)
                img = img[y_start:y_end,x_start:x_end]
                
            input_img = self.transform(img).unsqueeze(0)
            # print(input_img.shape)
            output = self.module(input_img)

            cam = self.drawCAM(self.img_width, self.img_height)

            pred_class = torch.argmax(output, dim=1)

            if self.transform_flag == 'ptf':
                img_cam = cv2.warpPerspective(cam, inverse_mat, (1280,720))
            elif self.transform_flag == 'crop':
                img_cam = np.zeros(original_img.shape[:2])
                img_cam[y_start:y_end,x_start:x_end] = cam
            else:
                img_cam = cam

            heatmap = cv2.applyColorMap(np.uint8(255 * img_cam), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            overlay = heatmap + np.float32(original_img / 255)
            overlay = overlay / np.max(overlay)

            cv2.putText(overlay, f"{curr_img_path} {self.classes[curr_img_label]} {self.curr_i}/{len(self.img_path)}",(10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0), 2)
            cv2.putText(overlay, f"Label: {self.classes[curr_img_label]} / Pred: {self.classes[pred_class]}",(10, 40),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0), 2)
            # for x, y in pts1:
            #     cv2.circle(overlay, (x, y), 10, (0, 255, 0), -1)
            cv2.imshow('overlay',overlay)
            cv2.imshow('img',img)

            pressed = cv2.waitKeyEx(15)
            if pressed == 27: break # Esc
            elif pressed == 56: self.change_curr_dirs(100) # 8
            elif pressed == 54: self.change_curr_dirs(1) # 6
            elif pressed == 52: self.change_curr_dirs(-1) # 4
            elif pressed == 50: self.change_curr_dirs(-100) # 2


username = getpass.getuser()
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', dest='path', required=True)
parser.add_argument('-m', '--model', dest='model', action = 'store')
parser.add_argument('-c', '--ckpt', dest='checkpoint', action = 'store')
parser.add_argument('-i', '--index', dest='index',type = int, default = 0)
parser.add_argument('-t', '--transform', dest='transform', action = 'store')
args = parser.parse_args()

torch.cuda.empty_cache()

viewer = Viewer(args.path,args.index,args.model, args.transform)   
viewer.view()
