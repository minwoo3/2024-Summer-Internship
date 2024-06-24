import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import convnext
from torchsummary import summary
import torch.nn.functional as  F
from CameraBlockDL.configs.templates import Commons
# vgg 중 사이즈 큰 애들 쓰고 싶을 때 쓰려고 가져온건데 굳이 그럴 필요 없을 것 같긴함
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

vgg_cfgs = {
    'beomjun_model_small': [64, 'M', 64, 'M', 64, 'M', 64, 'M'],     # 116k
    'beomjun_model_middle': [64, 'M', 128, 'M', 128, 'M', 256, 'M'], # 534k
    'beomjun_model_large': [64, 'M', 128, 'M', 256, 'M', 512, 'M'],  # 1.6M
    'beomjun_model_5': [64, 'M', 64, 'M', 128, 'M', 128, 'M', 256, 'M'], # 534k
    'no_pool': [64, 128, 128, 256],
    'one_pool': [64, 'M', 128, 128, 256],
    'two_pool': [64, 'M', 128, 128, 'M', 256],
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'C': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

"""
토치의 기본 nn.Module 객체 생성.
Pytorch Lighting 이 모듈을 감싸서 직접 사용하지는 않음.
pl_module.py 참조.
"""

class MaskNet(nn.Module):
    def __init__(self, num_of_classes=10, batch_norm=True, init_weights=False):
        super().__init__()
        self.shared_conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self.shared_bn1 = nn.BatchNorm2d(64)
        self.shared_conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.shared_bn2 = nn.BatchNorm2d(128)
        self.mask_conv1 = nn.Conv2d(128, 64, kernel_size=5, padding=2)
        self.mask_bn1 = nn.BatchNorm2d(64)
        self.mask_conv2 = nn.Conv2d(64, 1, kernel_size=5, padding=2)

        self.pred_conv1 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
        self.pred_bn1 = nn.BatchNorm2d(128)
        self.pred_conv2 = nn.Conv2d(128, 128, kernel_size=5, padding=2)

        self.linear1 = nn.Linear(128, 64)
        self.linear2 = nn.Linear(64, 10)

    def forward(self, x, return_mask=False):
        x = self.shared_bn1(F.relu(self.shared_conv1(x)))
        x = self.shared_bn2(F.relu(self.shared_conv2(x)))

        mask = self.mask_bn1(F.relu(self.mask_conv1(x)))
        mask = F.sigmoid(self.mask_conv2(mask))

        pred = self.pred_bn1(F.relu(self.pred_conv1(x)))
        pred = F.relu(self.pred_conv2(pred))

        mean = torch.mean(mask * pred, dim=[2, 3]) # dim = width, height (batch, channel, width, height)

        out = F.relu(self.linear1(mean))
        out = F.relu(self.linear2(out))
        if return_mask:
            return out, mask
        else:
            return out



class Net(nn.Module):
    def __init__(self, backbone, num_of_classes, batch_norm=False, init_weights=False, kernel_size=3, input_shape=(3, Commons.image.CROPPED_IMG_HEIGHT, Commons.image.CROPPED_IMG_WIDTH)):
        super().__init__()
        self.input_shape = input_shape
        self.features = self.make_layers(backbone, num_of_classes, batch_norm, kernel_size=kernel_size)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        # x = self.avgpool(x),
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def make_layers(self, backbone, num_of_classes, batch_norm = False, kernel_size=3):
        cfg = vgg_cfgs[backbone]
        layers=[]
        in_channels, h, w = self.input_shape
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                h, w = h//2, w//2
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=kernel_size, padding=(kernel_size-1)//2)
                if batch_norm:
                    # layers += [nn.Dropout(0.5), conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    # layers += [nn.Dropout(0.5), conv2d, nn.ReLU(inplace=True)]
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        # self.avgpool = nn.AdaptiveAvgPool2d((h, w))
        # layers.append(nn.Dropout(0.5))
        self.classifier = nn.Linear(in_channels * h * w, num_of_classes)
        return nn.Sequential(*layers)


def _model(backbone: str, num_of_classes: int, batch_norm: bool, pretrained: bool, progress:bool, input_shape=(3, Commons.image.CROPPED_IMG_HEIGHT, Commons.image.CROPPED_IMG_WIDTH), **kwargs) -> Net:
    if pretrained:
        kwargs['init_weights'] = False

    if backbone in ["mobilenet_v3_small"]:
        model = getattr(models, backbone)(pretrained=pretrained)
        # model = models.mobilenet_v2(pretrained=True)
        model.classifier = nn.Sequential(
            nn.Linear(
                in_features=model.classifier[0].in_features,
                out_features=num_of_classes
            )
        )
    elif backbone in ["mobilenet_v2", "efficientnet_b0"]:
        model = getattr(models, backbone)(pretrained=pretrained)
        model.classifier = nn.Sequential(
            nn.Linear(
                in_features=model.classifier[-1].in_features,
                out_features=num_of_classes
            )
        )
    elif backbone == "googlenet":
        model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)
        model.fc = nn.Linear(in_features=1024,
                             out_features=num_of_classes)
    elif backbone == "inception_v3":
        model = models.inception_v3(pretrained=True, progress=True)

    elif backbone == 'MaskNet':
        model = MaskNet()

    elif backbone == 'convnext':
        block_setting = [
            convnext.CNBlockConfig(64, 128, 3),
            convnext.CNBlockConfig(128, 256, 3),
        ]
        stochastic_depth_prob = 0.1
        model = convnext.ConvNeXt(block_setting, stochastic_depth_prob, num_classes=10)

    elif backbone == 'no_pool':
        model = Net(backbone=backbone, num_of_classes=num_of_classes, batch_norm=batch_norm, kernel_size=5, input_shape=input_shape,**kwargs)
    else:
        # 개인적으로 만든 모델들은 pretrain없이 학습
        model = Net(backbone=backbone,num_of_classes=num_of_classes,batch_norm=batch_norm, input_shape=input_shape, **kwargs)

    return model

if __name__ == "__main__":
    # model = _model(backbone='googlenet', num_of_classes=10, batch_norm=True, pretrained=False, progress=False)
    # summary(model, (3, 44, 60), batch_size=10, device='cpu')
    # model = _model('no_pool',  num_of_classes=10, batch_norm=True, pretrained=False, progress=False)
    # summary(model, (3, 44, 60), batch_size=10, device='cpu')
    block_setting = [
        convnext.CNBlockConfig(64, 128, 3),
        convnext.CNBlockConfig(128, 256, 3),
    ]

    num_of_classes = 13

    stochastic_depth_prob = 0.1
    model = convnext.ConvNeXt(block_setting, stochastic_depth_prob, num_classes=num_of_classes)
    summary(model, (3, 44, 60), batch_size=10, device='cpu')
    print()

    model_m = _model('beomjun_model_middle', num_of_classes=num_of_classes, batch_norm=True, pretrained=False, progress=False)
    summary(model_m, (3, 44, 60), batch_size=10, device='cpu')

    print()
    model_l = _model('beomjun_model_large', num_of_classes=num_of_classes, batch_norm=True, pretrained=False, progress=False)
    summary(model_l, (3, 44, 60), batch_size=10, device='cpu')
