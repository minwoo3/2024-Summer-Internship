import csv
import cv2
import torch
import math
import sys, os
import argparse
import getpass
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
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
        with open(csv_path,'r',newline='') as f:
            data = list(csv.reader(f))

        self.img_path, self.img_label= [], []
        for path, label in data:
            self.img_path.append(path)
            self.img_label.append(int(label))

        self.classes = ['clean','dirty']
        opt, batch_size = 1e-5, 16
        # datamodule = RoadStadusDataModule(batch_size = batch_size, transform_flag = self.transform_flag)
        # datamodule.setup(stage='fit')

        # example_img, _, _ = datamodule.train_dataset[0]
        # self.img_height, self.img_width = example_img.shape[-2:]  # (height, width)
        self.img_height, self.img_width = 720, 1280  # (height, width)
        if model in ['cnn','CNN']:
            self.module = CNNModule.load_from_checkpoint('/home/rideflux/2024-Summer-Internship/RoadStatusModel/test/CNNModule_2024-07-18_epochs_20.ckpt',
                                                    img_width=self.img_width, 
                                                    img_height=self.img_height, 
                                                    opt=opt)
        elif model in ['resnet','res','ResNet']:
            self.module = ResnetModule.load_from_checkpoint(f'{ssd_dir}/checkpoint/CNNModule_2024-07-16_epochs_20.ckpt',opt)
        else:
            raise ValueError("Invalid model name. Choose from ['cnn', 'CNN', 'resnet', 'res', 'ResNet']")
    
        self.module_name = self.module.__class__.__name__
        
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

    def change_curr_dirs(self, dif):
        self.curr_i += dif
        if self.curr_i == len(self.img_path):
            print('end of list')
            self.curr_i -= dif
        elif self.curr_i == 0:
            print('first of list')
            self.curr_i += dif

    def drawCAM(self, argmax, width, height):
        activation_map = self.module.featuremap.squeeze().cpu()
        # gap 방식
        class_weights_gap = F.adaptive_avg_pool2d(activation_map,(1,1)).squeeze()
        cam = torch.zeros(activation_map.shape[1:], dtype=torch.float32)
        for i in range(len(class_weights_gap)):
            cam += class_weights_gap[i]*activation_map[i,:,:]

        params = list(self.module.parameters())
        weight_softmax = params[-2].cpu()
        num_channels = activation_map.size(0)
        class_weights = weight_softmax[argmax.item()].view(num_channels, math.floor(height/32), math.floor(width/32))
        cam = torch.zeros(activation_map.shape[1:], dtype=torch.float32)
        for i in range(len(class_weights)):
            cam += class_weights[i,:,:]*activation_map[i,:,:]

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
            img = cv2.imread(curr_img_path)

            if self.transform_flag == 'ptf':
                img = cv2.resize(img,(1280, 720))
                pts1 = [[330, 500], [950, 500], [100, 650], [1200, 650]]
                img, transform_mat = ptf(pts1,img)
                inverse_mat = np.linalg.inv(transform_mat)
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            input_img = self.transform(img).unsqueeze(0)
            output = self.module(input_img)
            _, argmax = torch.max(output, 1)

            img_cam = self.drawCAM(argmax, self.img_width, self.img_height)

            pred_class = torch.argmax(output, dim=1)

            if self.transform_flag == 'ptf':
                img_cam = cv2.warpPerspective(img_cam, inverse_mat, (1280,720))

            heatmap = cv2.applyColorMap(np.uint8(255 * img_cam), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            img = np.array(img)
            overlay = heatmap + np.float32(img / 255)
            overlay = overlay / np.max(overlay)

            cv2.putText(overlay, f"{curr_img_path} {self.classes[curr_img_label]} {self.curr_i}/{len(self.img_path)}",(10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0), 2)
            cv2.putText(overlay, f"Label: {self.classes[curr_img_label]} / Pred: {self.classes[pred_class]}",(10, 40),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0), 2)
            # for x, y in pts1:
            #     cv2.circle(overlay, (x, y), 10, (0, 255, 0), -1)
            cv2.imshow('overlay',overlay)

            pressed = cv2.waitKeyEx(15)
            if pressed == 27: break # Esc
            elif pressed == 56: self.change_curr_dirs(100) # 8
            elif pressed == 54: self.change_curr_dirs(1) # 6
            elif pressed == 52: self.change_curr_dirs(-1) # 4
            elif pressed == 50: self.change_curr_dirs(-100) # 2


username = getpass.getuser()
ssd_dir = f'/media/{username}/T7/2024-Summer-Internship'
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', dest='path', required=True)
parser.add_argument('-m', '--model', dest='model', action = 'store')
parser.add_argument('-i', '--index', dest='index',type = int, default = 0)
parser.add_argument('-t', '--transform', dest='transform', action = 'store')
args = parser.parse_args()

torch.cuda.empty_cache()

viewer = Viewer(args.path,args.index,args.model, args.transform)   
viewer.view()
