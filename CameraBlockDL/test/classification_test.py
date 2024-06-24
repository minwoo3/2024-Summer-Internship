# TODO. TP, FP, TN, FN 계산하는 코드 만들기

# TODO. 테스트 셋에 대한 평가 진행
import sys
sys.path.append('/home/ampere_2way_2/Perception-CodeVault')
import pytorch_lightning as pl
import torch
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import os
from CameraBlockDL.data.dataset import get_loader, get_blockdataset
from CameraBlockDL.model.pl_module import get_module
from CameraBlockDL.data.dataset import DataModule
from CameraBlockDL.miscs.pformat import pprint
from CameraBlockDL.configs.config import save_cfg, load_cfg, create_config
import cv2
import subprocess
from torchsummary import summary

def python_test(cfg, verbose=True):
    torch.manual_seed(cfg.seed)
    trainer = pl.Trainer()
    module = get_module(cfg)
    # loader = get_loader(cfg, dsname=cfg.dataset_names.split(',')[0], dstype="test")
    # result = trainer.test(module, dataloaders=loader)
    dm = DataModule(cfg)
    result = trainer.test(module, dm)
    if verbose:
        pprint(result, ["OKBLUE"])

    del trainer
    del module
    del dm

def new_data_test(model_cfg, data_cfg):
    torch.manual_seed(model_cfg.seed)
    trainer = pl.Trainer()
    module = get_module(model_cfg)
    dm = DataModule(data_cfg)
    result = trainer.test(module, dm.test_dataloader()[0])
    pprint(result, ["OKBLUE"])

def single_image_test(model_cfg, img_src):
    ############## settings ##############
    cfg = load_cfg(model_cfg)
    visualize = True
    img_tf = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ######################################

    model = get_module(cfg)
    model.cuda()
    model.eval()
    CROPPED_IMG_WIDTH = 120
    CROPPED_IMG_HEIGHT = 88
    # summary(model, (3, CROPPED_IMG_HEIGHT, CROPPED_IMG_WIDTH), batch_size=10)
    cvimg = cv2.imread(f'{img_src}')

    height, width, channel = cvimg.shape
    pad_h = height * 3 // 6
    pad_w = width * 3 // 6
    padded_img = np.zeros((height + pad_h, width + pad_w, channel), dtype=np.uint8)
    padded_img[:height, pad_w:, :] = cvimg.copy()
    cvimg = padded_img
    cvimg = cv2.resize(cvimg, dsize=(CROPPED_IMG_WIDTH, CROPPED_IMG_HEIGHT))
    if visualize:
        cv2.imwrite(f'padded_resize_{CROPPED_IMG_WIDTH}_{CROPPED_IMG_HEIGHT}.png', cvimg)

    npimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
    npimg = np.swapaxes(npimg, 0, 1)
    npimg = np.swapaxes(npimg, 0, 2)
    with torch.no_grad():
        ptimg = torch.from_numpy(npimg / 255)
        ptimg = img_tf(ptimg).type(torch.FloatTensor)
        ptimg = ptimg.unsqueeze(0).cuda()
        out_logits = model(ptimg).cpu().numpy().squeeze()
        print(out_logits)



def cpp_test(cfg, verbose=True):
    ...
    result = subprocess.check_output(...)
    if verbose:
        pprint(result, ["OKBLUE"])

if __name__=='__main__':
    cfgs = ['2023-08-21-08-55-36_beomjun_model_middle_PER-370-44-60-0',
            '2023-08-18-18-16-49_beomjun_model_middle_PER-370-3']

    single_image_test(cfgs[0],
                      '/home/ampere_2way_2/Perception-CodeVault/TrafficlightClassificationDL/test/test_image_6.png')