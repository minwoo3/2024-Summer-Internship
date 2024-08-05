import torch.nn as nn
import torch.nn.functional as F
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
        self.fc = nn.Linear(self.feature_c*self.feature_w*self.feature_h, 2)
    
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


############ ResNet50 ###############
def conv_block(in_dim, out_dim, kernel_size, activation, stride=1):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride),
        nn.BatchNorm2d(out_dim),
        activation,
    )
    return model

class BottleNeck(nn.Module):
    def __init__(self,in_dim,mid_dim,out_dim,activation,down=False):
        super(BottleNeck,self).__init__()
        self.down=down
        # 피처맵의 크기가 감소하는 경우
        if self.down:
            self.layer = nn.Sequential(
              conv_block(in_dim,mid_dim,1,activation,stride=2),
              conv_block(mid_dim,mid_dim,3,activation,stride=1),
              conv_block(mid_dim,out_dim,1,activation,stride=1),
            )
            
            # 피처맵 크기 + 채널을 맞춰주는 부분
            self.downsample = nn.Conv2d(in_dim,out_dim,kernel_size=1,stride=2)
            
        # 피처맵의 크기가 그대로인 경우
        else:
            self.layer = nn.Sequential(
                conv_block(in_dim,mid_dim,1,activation,stride=1),
                conv_block(mid_dim,mid_dim,3,activation,stride=1),
                conv_block(mid_dim,out_dim,1,activation,stride=1),
            )
        self.dim_equalizer = nn.Conv2d(in_dim, out_dim, kernel_size=1)
    
    def forward(self,x):
        if self.down:
            downsample = self.downsample(x)
            out = self.layer(x)
            out = out + downsample
        else:
            out = self.layer(x)
            if x.size() is not out.size():
                x = self.dim_equalizer(x)
            out = out + x
        return out

class ResNet50(nn.Module):
    def __init__(self, base_dim, num_classes=10):
        super(ResNet50, self).__init__()
        self.activation = nn.ReLU()
        self.layer_1 = nn.Sequential(
            nn.Conv2d(3,base_dim,7,2,3),
            nn.ReLU(),
            nn.MaxPool2d(3,2,1),
        )
        self.layer_2 = nn.Sequential(
            BottleNeck(base_dim,base_dim,base_dim*4,self.activation),
            BottleNeck(base_dim*4,base_dim,base_dim*4,self.activation),
            BottleNeck(base_dim*4,base_dim,base_dim*4,self.activation,down=True),
        )   
        self.layer_3 = nn.Sequential(
            BottleNeck(base_dim*4,base_dim*2,base_dim*8,self.activation),
            BottleNeck(base_dim*8,base_dim*2,base_dim*8,self.activation),
            BottleNeck(base_dim*8,base_dim*2,base_dim*8,self.activation),
            BottleNeck(base_dim*8,base_dim*2,base_dim*8,self.activation,down=True),
        )
        self.layer_4 = nn.Sequential(
            BottleNeck(base_dim*8,base_dim*4,base_dim*16,self.activation),
            BottleNeck(base_dim*16,base_dim*4,base_dim*16,self.activation),
            BottleNeck(base_dim*16,base_dim*4,base_dim*16,self.activation),            
            BottleNeck(base_dim*16,base_dim*4,base_dim*16,self.activation),
            BottleNeck(base_dim*16,base_dim*4,base_dim*16,self.activation),
            BottleNeck(base_dim*16,base_dim*4,base_dim*16,self.activation,down=True),
        )
        self.layer_5 = nn.Sequential(
            BottleNeck(base_dim*16,base_dim*8,base_dim*32,self.activation),
            BottleNeck(base_dim*32,base_dim*8,base_dim*32,self.activation),
            BottleNeck(base_dim*32,base_dim*8,base_dim*32,self.activation),
        )
        self.avgpool = nn.AvgPool2d(1,1) 
        self.fc_layer = nn.Linear(base_dim*32,num_classes)
        
    def forward(self, x, train_batch_sz):
        out = self.layer_1(x)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = self.layer_5(out)
        out = self.avgpool(out)
        out = out.view(train_batch_sz,-1)
        out = self.fc_layer(out)
        return out