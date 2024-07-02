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
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from data.RoadStatusModelDS_v2 import RoadStatusDataset
from torchvision.transforms.functional import to_pil_image

def drawCAM(img,argmax):
    global model
    # output = model(input_img)
    # _, argmax = torch.max(output, 1)
    activation_map = model.featuremap.squeeze().cpu()
    # print(activation_map.shape)
    params = list(model.parameters())
    weight_softmax = params[-2].cpu()
    # print(weight_softmax.shape)
    class_weights = weight_softmax[argmax.item()].view(128, 22, 40)
    cam = torch.zeros(activation_map.shape[1:], dtype=torch.float32)
    for i in range(len(class_weights)):
        cam += class_weights[i,:,:]*activation_map[i,:,:]
    cam = F.relu(cam)
    cam = cam - cam.min()
    cam = cam / cam.max()
    cam = cam.detach().numpy()
    cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((to_pil_image(img).width, to_pil_image(img).height), Image.Resampling.LANCZOS)) / 255.0
    return cam_resized


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
    front_path = '/media/ampere_2_1/T7/2024-Summer-Internship'
else:
    front_path = '/media/rideflux/T7/2024-Summer-Internship'
    
# dirty_ds = DirtyRoadDataset(front_path + '/2024-Summer-Internship/CameraBlockDL/data/dirty.csv')
clean_test_ds = RoadStatusDataset(front_path + '/scene/clean_test.csv')
dirty_test_ds = RoadStatusDataset(front_path + '/scene/dirty_test.csv')

# Total dataset 만들기 = dirty + clean
test_ds = dirty_test_ds + clean_test_ds

# DataLoader 선언
test_dl = DataLoader(dataset = test_ds, batch_size = test_batch_sz, shuffle= True, drop_last = False)

classes = ['clean', 'dirty']
model = torch.load('/home/rideflux/2024-Summer-Internship/CameraBlockDL/test/1719907458.2549872_model.pt')
model.eval()
cols, rows = 8, 4
img_list = os.listdir('/home/rideflux/2024-Summer-Internship/CameraBlockDL/test/image')
img_cnt = len(img_list)

for i in range(1, cols*rows + 1):
    img_list = os.listdir('/home/rideflux/2024-Summer-Internship/CameraBlockDL/test/image')
    img_cnt = len(img_list)
    data_index = np.random.randint(len(test_ds))
    img = test_ds[data_index][0]
    input_img = img.unsqueeze(dim=0).to("cuda")
    output = model(input_img)
    _, argmax = torch.max(output, 1)
    pred = classes[argmax.item()]
    label = classes[test_ds[data_index][1]]
    cam_resized = drawCAM(img,argmax)

    plot_img = test_ds[data_index][0].permute(1,2,0)
    plt.subplot(2,1,1)
    plt.imshow(plot_img)
    plt.title('original img')

    plt.subplot(2,1,2)
    plt.title(f'{label}/{pred}')
    plt.imshow(plot_img)
    plt.imshow(cam_resized, cmap = 'jet', alpha = 0.5)
    # plt.axis('off')
    plt.savefig(f'/home/rideflux/2024-Summer-Internship/CameraBlockDL/test/image/{img_cnt}.jpg', format = 'jpeg')
