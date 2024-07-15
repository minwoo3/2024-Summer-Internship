import torch
import torch.nn.functional as F
import numpy as np
import sys, os
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.RoadStatusModelNN import CNNModel

# 모델 준비
model = torch.load('resultmodel.pt')
model.eval()

label_map = ['clean', 'dirty']

# 입력 이미지 준비
preprocess = transforms.Compose([
    transforms.Resize((180, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img_path = '/home/rideflux/2024-Summer-Internship/CameraBlockDL/test/다운로드.jpeg'
img = Image.open(img_path).convert('RGB')
input_tensor = preprocess(img).unsqueeze(0)

# 예측 수행 및 CAM 생성
output = model(input_tensor.to('cuda'))
pred_class = output.argmax(dim=1).item()
_, argmax = torch.max(output, 1)
pred = label_map[argmax.item()]
label = label_map[pred_class]
# 마지막 컨볼루션 레이어의 활성화 가져오기
activation_map = model.featuremap.squeeze().cpu()
# print(activation_map.shape)

# 마지막 레이어의 가중치 가져오기
params = list(model.parameters())
# print('params:',len(params), len(params[0]), len(params[0][0]))
weight_softmax = params[-2].cpu()

# 예측된 클래스의 가중치
class_weights = weight_softmax[pred_class].view(128,5,10)
# print('class weight shape:',class_weights.shape)
# CAM 계산
cam = torch.zeros(activation_map.shape[1:], dtype=torch.float32)
# print(cam.shape)
for i in range(len(class_weights)):
    cam += class_weights[i,:,:] * activation_map[i, :, :]

# CAM 정규화 및 원본 이미지 크기로 업샘플링
cam = F.relu(cam)
cam = cam - cam.min()
cam = cam / cam.max()
cam = cam.detach().numpy()
cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((img.width, img.height), Image.Resampling.LANCZOS)) / 255.0

# CAM 시각화
plt.imshow(img)
plt.imshow(cam_resized, cmap='jet', alpha=0.5)
plt.axis('off')
plt.title(f'Predicted : {label}, {pred}')
plt.show()