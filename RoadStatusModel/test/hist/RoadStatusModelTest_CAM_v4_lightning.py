import csv
import torch
import math
import sys, os
import argparse
import getpass
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.RoadStatusModelNN_lightning import CNNModule, ResnetModule

def drawCAM(argmax, width, height):
    global model
    activation_map = model.featuremap.squeeze().cpu()
    params = list(model.parameters())
    weight_softmax = params[-2].cpu()
    num_channels = activation_map.size(0)
    print(num_channels)
    if model.__class__.__name__ == 'CNNModule':
        class_weights = weight_softmax[argmax.item()].view(num_channels, math.floor(height/32), math.floor(width/32))
    elif model.__class__.__name__ == 'ResNetModule':
        class_weights = weight_softmax[argmax.item()].view(num_channels, 1, 1)
        class_weights = class_weights.expand(num_channels, activation_map.size(1), activation_map.size(2))
    cam = torch.zeros(activation_map.shape[1:], dtype=torch.float32)
    for i in range(len(class_weights)):
        cam += class_weights[i,:,:]*activation_map[i,:,:]
    cam = F.relu(cam)
    cam = cam - cam.min()
    cam = cam / cam.max()
    cam = cam.detach().numpy()
    cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((width, height), Image.Resampling.LANCZOS)) / 255.0
    return cam_resized

def csvreader(csv_dir):
    with open(csv_dir, 'r', newline='') as f:
        data = list(csv.reader(f))
    paths, labels = [], []
    for path, label in data:
        paths.append(path)
        labels.append(label)
    print('Read CSV Successfully')
    return paths, labels

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--crop', dest='crop', action = 'store_true')
args = parser.parse_args()

username = getpass.getuser()
img_width, img_height = 1280, 720
paths, labels = csvreader(f'/media/{username}/T7/2024-Summer-Internship/scene/test.csv')
model = CNNModule.load_from_checkpoint('CNNModule_epochs_20_lr_1e-05_crop.ckpt', opt = 1e-5, img_width = img_width, img_height = img_height)
# model = ResnetModule.load_from_checkpoint('ResnetModule_epochs_20_lr_1e-05.ckpt', opt = 1e-5, strict=False)
model.eval()

if args.crop:
    transform = transforms.Compose([
        transforms.Resize((720, 1280)),
        transforms.Lambda(lambda img: img.crop((0, int(img.height*0.5), img.width, img_height))),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
    ])
else:
    transform = transforms.Compose([
        transforms.Resize((720, 1280)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
    ])

classes = ['Clean', 'Dirty']
showing_img_cnt = len(paths)
now_img_cnt = 0

# def update_image_list(now_img_cnt):
#     global false_image, showing_img_cnt, nas_dir
#     # nas_dir = f'/home/ampere_2_1/GeneralCase/Raw/{scene_num}/camera_0'
#     # false_image = [os.path.join(nas_dir, f) for f in os.listdir(nas_dir)]
#     showing_img_cnt = len(false_image)

def show_image(index):
    global now_img_cnt, img_width, img_height, classes
    now_img_cnt = index
    path, label = paths[now_img_cnt], int(labels[now_img_cnt])
    
    img = Image.open(path)
    img_t = transform(img)
    output = model(img_t.unsqueeze(dim=0))
    _, argmax = torch.max(output, 1)
    img_cam = drawCAM(argmax, int(img_width), int(img_height/2))
    pred = classes[argmax.item()]
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f'{now_img_cnt + 1}/{showing_img_cnt}, Path: {path}', fontsize=12, color='blue', weight='bold')

    axs[0].imshow(img)
    axs[0].set_title(f'origin: {label}')
    axs[0].axis('off')

    axs[1].imshow(img_t.permute(1,2,0))
    axs[1].imshow(img_cam, cmap='jet', alpha=0.5)
    # axs[1].imshow(cam_on_original, cmap='jet', alpha=0.5)
    axs[1].set_title(f'predict: {pred}')
    axs[1].axis('off')

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

def on_key(event):
    global now_img_cnt, scene_num
    if event.key == 'right':  # Next image
        if now_img_cnt < showing_img_cnt - 1:
            now_img_cnt += 1
            plt.close()
            show_image(now_img_cnt)
    elif event.key == 'left':  # Previous image
        if now_img_cnt > 0:
            now_img_cnt -= 1
            plt.close()
            show_image(now_img_cnt)
    elif event.key == 'up':  # Next Scene
        if now_img_cnt < showing_img_cnt - 100:
            now_img_cnt += 100
            plt.close()
            show_image(now_img_cnt)
    elif event.key == 'down':  # Previous Scene
        if now_img_cnt > 100:
            now_img_cnt -= 100
            plt.close()
            show_image(now_img_cnt)
    elif event.key == 'q':  # Quit
        plt.close('all')

show_image(now_img_cnt)
