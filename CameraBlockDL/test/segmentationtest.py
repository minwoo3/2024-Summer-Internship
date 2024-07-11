import torch
from torchvision import transforms
from segmentation_models_pytorch import Unet
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# UNet 모델 초기화
model = Unet(encoder_name="resnet34",        # 선택할 수 있는 여러 encoder가 있습니다
             encoder_weights="imagenet",     # ImageNet 사전 훈련된 가중치 사용
             in_channels=3,                  # 입력 채널 수 (RGB 이미지이므로 3)
             classes=1)                      # 출력 채널 수 (세그멘테이션 클래스 수)

# 예제 이미지 로드 및 전처리
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 크기를 더 크게 조정
    transforms.ToTensor(),
])

image_path = '49.jpg'
image = Image.open(image_path)
image = transform(image).unsqueeze(0)

# 모델 추론
model.eval()
with torch.no_grad():
    output = model(image)

# 결과 시각화
output = torch.sigmoid(output).squeeze().cpu().numpy()

# 시각화를 위해 원본 이미지와 결과를 함께 표시
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(Image.open(image_path))
axs[0].set_title('Original Image')
axs[0].axis('off')

axs[1].imshow(output, cmap='gray')
axs[1].set_title('Segmentation Output')
axs[1].axis('off')

plt.show()
