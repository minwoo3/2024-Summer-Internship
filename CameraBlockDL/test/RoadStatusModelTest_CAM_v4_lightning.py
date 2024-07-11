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
    # class_weights = weight_softmax[argmax.item()].view(num_channels,  activation_map.size(1), activation_map.size(2))
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
    for i in range(len(data)):
        data[i] = ''.join(data[i])
    print('Read CSV Successfully')
    print(f'{data[0]} ...')
    return data

def txtreader(txt_dir):
    with open(txt_dir, 'r', newline='') as file:
        data = file.readlines()
        data = [path.rstrip('\n') for path in data]
    print('Read txt Successfully')
    print(f'{data[0]} ...')
    return data

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--crop', dest='crop', action = 'store_true')
args = parser.parse_args()

if args.crop:
    model_img_width, model_img_height = 1280, 720*0.4
    transform = transforms.Compose([
        transforms.Resize((720, 1280)),
        transforms.Lambda(lambda img: img.crop((0, int(img.height*0.5), img.width, int(img.height*0.9)))),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
    ])
else:
    model_img_width, model_img_height = 1280, 720
    transform = transforms.Compose([
        transforms.Resize((720, 1280)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
    ])

ssd_dir = f'/media/{getpass.getuser()}/T7/2024-Summer-Internship/scene'
ds_list = os.listdir(ssd_dir)
test_scene = []
for ds in ds_list:
    if 'test' in ds:
        scenes = csvreader(ssd_dir+f'/{ds}')
        for image in scenes:
            test_scene.append(image)

# model = CNNModule.load_from_checkpoint('CNNModule_epochs_20_lr_1e-05.ckpt', opt = 1e-5, img_width = model_img_width, img_height = model_img_height)
model = ResnetModule.load_from_checkpoint('ResnetModule_epochs_20_lr_1e-05.ckpt', opt = 1e-5, strict = False)
model.eval()
classes = ['Clean', 'Dirty']
showing_img_cnt = len(test_scene)
now_img_cnt = 0

def show_image(index):
    global now_img_cnt
    global model_img_width, model_img_height
    now_img_cnt = index
    path = test_scene[now_img_cnt]
    img = Image.open(path)
    # img = img.resize((1280,720))
    width, height = img.width, img.height
    img_t = transform(img)
    output = model(img_t.unsqueeze(dim=0))
    _, argmax = torch.max(output, 1)
    img_cam = drawCAM(argmax, width, height)
    # cam_on_original = np.zeros((height, width))
    # cam_on_original[int(height*0.5):int(height*0.9), 0:width] = img_cam
    pred = classes[argmax.item()]
    if 'NIA' in path:
        label = 'Clean'
    else:
        label = 'Dirty'
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f'{now_img_cnt + 1}/{showing_img_cnt}, Path: {path}', fontsize=12, color='blue', weight='bold')

    axs[0].imshow(img)
    axs[0].set_title(f'origin: {label}')
    # axs[0].axis('off')

    axs[1].imshow(img)
    axs[1].imshow(img_cam, cmap='jet', alpha=0.5)
    # axs[1].imshow(cam_on_original, cmap='jet', alpha=0.5)
    axs[1].set_title(f'predict: {pred}')
    # axs[1].axis('off')

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

def on_key(event):
    global now_img_cnt
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
    if event.key == 'up':  # Next image
        if now_img_cnt < showing_img_cnt - 500:
            now_img_cnt += 500
            plt.close()
            show_image(now_img_cnt)
    elif event.key == 'down':  # Previous image
        if now_img_cnt > 500:
            now_img_cnt -= 500
            plt.close()
            show_image(now_img_cnt)
    elif event.key == 'q':  # Quit
        plt.close('all')

show_image(now_img_cnt)
