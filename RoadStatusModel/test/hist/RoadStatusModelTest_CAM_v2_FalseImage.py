import cv2
import torch
import sys, os
import argparse
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
import torchvision.utils
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from data.RoadStatusModelDS_v2 import RoadStatusDataset
from torchvision.transforms.functional import to_pil_image

def drawCAM(img,argmax):
    global model
    activation_map = model.featuremap.squeeze().cpu()
    params = list(model.parameters())
    weight_softmax = params[-2].cpu()
    class_weights = weight_softmax[argmax.item()].view(128,5,10)
    cam = torch.zeros(activation_map.shape[1:], dtype=torch.float32)
    for i in range(len(class_weights)):
        cam += class_weights[i,:,:]*activation_map[i,:,:]
    cam = F.relu(cam)
    cam = cam - cam.min()
    cam = cam / cam.max()
    cam = cam.detach().numpy()
    cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((to_pil_image(img).width, to_pil_image(img).height), Image.Resampling.LANCZOS)) / 255.0
    return cam_resized

transform = transforms.Compose([
            transforms.Resize((180, 320)),
            transforms.ToTensor()])
clean_train_ratio, clean_val_ratio = 0.7, 0.15
dirty_train_ratio, dirty_val_ratio = 0.5, 0.5
train_batch_sz, val_batch_sz, test_batch_sz = 32, 2, 20
epochs, learning_rate = 10, 1e-5


imgs = ['/media/ampere_2_1/T7/2024-Summer-Internship/벚꽃/04-04/2024-04-04-17-35-51_solati_v5_6_397-418/camera_0/107.jpg', '/media/ampere_2_1/T7/2024-Summer-Internship/벚꽃/04-04/2024-04-04-17-35-51_solati_v5_6_397-418/camera_0/2.jpg', '/media/ampere_2_1/T7/2024-Summer-Internship/벚꽃/04-04/2024-04-04-17-35-51_solati_v5_6_397-418/camera_0/3.jpg', '/media/ampere_2_1/T7/2024-Summer-Internship/벚꽃/04-03/2024-04-03-13-07-59_solati_v5_6_646-667/camera_0/43.jpg', '/media/ampere_2_1/T7/2024-Summer-Internship/벚꽃/04-04/2024-04-04-17-35-51_solati_v5_6_397-418/camera_0/5.jpg', '/media/ampere_2_1/T7/2024-Summer-Internship/벚꽃/04-03/2024-04-03-13-07-59_solati_v5_6_646-667/camera_0/73.jpg', '/media/ampere_2_1/T7/2024-Summer-Internship/벚꽃/04-04/2024-04-04-17-35-51_solati_v5_6_397-418/camera_0/0.jpg', '/media/ampere_2_1/T7/2024-Summer-Internship/벚꽃/04-04/2024-04-04-17-35-51_solati_v5_6_397-418/camera_0/1.jpg', '/media/ampere_2_1/T7/2024-Summer-Internship/벚꽃/04-04/2024-04-04-17-35-51_solati_v5_6_397-418/camera_0/4.jpg', '/media/ampere_2_1/T7/2024-Summer-Internship/벚꽃/04-09/2024-04-09-16-28-08_solati_v5_6_62-83/camera_0/137.jpg', '/media/ampere_2_1/T7/2024-Summer-Internship/벚꽃/04-09/2024-04-09-16-28-08_solati_v5_6_62-83/camera_0/138.jpg', '/media/ampere_2_1/T7/2024-Summer-Internship/벚꽃/04-04/2024-04-04-17-35-51_solati_v5_6_397-418/camera_0/108.jpg', '/media/ampere_2_1/T7/2024-Summer-Internship/벚꽃/04-09/2024-04-09-14-39-26_solati_v5_6_703-724/camera_0/0.jpg', '/media/ampere_2_1/T7/2024-Summer-Internship/벚꽃/04-04/2024-04-04-17-35-51_solati_v5_6_397-418/camera_0/6.jpg', '/media/ampere_2_1/T7/2024-Summer-Internship/벚꽃/04-09/2024-04-09-16-28-08_solati_v5_6_62-83/camera_0/136.jpg']

classes = ['clean', 'dirty']
model = torch.load('resultmodel.pt')
model.eval()
cols, rows = 8, 4
for i in range(1, cols*rows + 1):
    img_list = os.listdir('/home/ampere_2_1/Seojeonghyun/2024-Summer-Internship/CameraBlockDL/test/FalseImage')
    img_cnt = len(img_list)
    data_index = np.random.randint(len(imgs))
    img = Image.open(imgs[data_index])
    img = transform(img)
    input_img = img.unsqueeze(dim=0).to("cuda")
    output = model(input_img)
    _, argmax = torch.max(output, 1)
    pred = classes[argmax.item()]
    cam_resized = drawCAM(img,argmax)

    plt.subplot(2,1,1)
    plt.imshow(img.permute(1,2,0))
    plt.title('original img')

    plt.subplot(2,1,2)
    plt.title(f'dirty/{pred}')
    plt.imshow(img.permute(1,2,0))
    plt.imshow(cam_resized, cmap = 'jet', alpha = 0.5)
    # plt.axis('off')
    plt.savefig(f'/home/ampere_2_1/Seojeonghyun/2024-Summer-Internship/CameraBlockDL/test/FalseImage/{img_cnt}.jpg', format = 'jpeg')
