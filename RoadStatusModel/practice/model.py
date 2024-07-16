import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, img_width, img_height):
        super(CNNModel, self).__init__()
        self.featuremap = None
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # 임의의 입력 텐서로 fc 레이어의 입력 크기 계산
        self.fc_input_dim = self._get_fc_input_dim(img_width, img_height)
        print(self.fc_input_dim)
        self.fc = nn.Linear(self.fc_input_dim, 2)
        
    def _get_fc_input_dim(self, img_width, img_height):
        # 임의의 입력 텐서 생성 (배치 크기는 1로 설정)
        x = torch.randn(1, 3, img_height, img_width)
        # sequential 네트워크를 통과시킴
        x = self.sequential(x)
        # flatten 할 때의 크기를 반환 (즉, 요소의 총 개수)
        return x.numel()
        
    def forward(self, x):
        x = self.sequential(x)
        self.featuremap = x
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.softmax(x, dim=1)
        
        return x

# 모델 생성 예시
img_width, img_height = 1280, 720
model = CNNModel(img_width, img_height)

