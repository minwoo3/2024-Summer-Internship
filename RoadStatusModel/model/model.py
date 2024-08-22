import torch.nn as nn
import torch

class CNNModel(nn.Module):
    def __init__(self, img_width, img_height):
        super(CNNModel, self).__init__()
        self.featuremap = None
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, padding=1), 
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
        self._get_fc_input_dim(img_width,img_height)
        self.fc = nn.Linear(self.feature_c*self.feature_w*self.feature_h, 1)
    
    def _get_fc_input_dim(self, img_width, img_height):
        x = torch.randn(1,4,img_width,img_height)
        x = self.sequential(x)
        self.feature_c, self.feature_w, self.feature_h = x.size(1), x.size(2), x.size(3)

    def forward(self, x):
        x = self.sequential(x)
        self.featuremap = x
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x