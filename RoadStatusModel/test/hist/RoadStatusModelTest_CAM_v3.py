import csv
import torch
import sys, os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

def drawCAM(argmax,width,height):
    global model
    activation_map = model.featuremap.squeeze().cpu()
    params = list(model.parameters())
    weight_softmax = params[-2].cpu()
    class_weights = weight_softmax[argmax.item()].view(128, 22, 40)
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
    with open(csv_dir,'r',newline='') as f:
        data = list(csv.reader(f))
    for i in range(len(data)):
        data[i] = ''.join(data[i])
    print('Read CSV Successfully')
    print(f'{data[0]} ...')
    return data

def txtreader(txt_dir):
    with open(txt_dir,'r',newline='') as file:
        data = file.readlines()
        data = [path.rstrip('\n') for path in data]
    print('Read txt Successfully')
    print(f'{data[0]} ...')
    return data
    
transform = transforms.Compose([
            transforms.Resize((720, 1280)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
        ])

false_image = txtreader('result.txt')
model = torch.load('1719907458.2549872_model.pt')
model.eval()
classes = ['Clean', 'Dirty']
showing_img_cnt = len(false_image)
now_img_cnt = 0
for path in false_image:
    now_img_cnt += 1
    img = Image.open(path)
    width, height = img.width, img.height
    img_t = transform(img)
    output = model(img_t.unsqueeze(dim=0).to('cuda'))
    _, argmax = torch.max(output, 1)
    img_cam = drawCAM(argmax, width, height)
    pred = classes[argmax.item()]
    if 'NIA' in path:
        label = 'Clean'
    else:
        label = 'Dirty'
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f'{now_img_cnt}/{showing_img_cnt}, Path: {path}', fontsize=12, color='blue', weight='bold')

    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.title(f'origin: {label}')

    plt.subplot(1,2,2)
    plt.imshow(img)
    plt.imshow(img_cam, cmap='jet', alpha = 0.5)
    plt.title(f'predict: {pred}')
    plt.show()
