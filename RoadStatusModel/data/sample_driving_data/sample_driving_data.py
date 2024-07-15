import os
import random
import cv2
import numpy as np
import torch
from hyunkun.tl_dataset_update.utils import jpg2pt
dir_path = os.path.dirname(os.path.realpath(__file__))

"""
v4, v5 차량 플랫폼의 주행 이미지 중에 샘플링 하기 위한 코드
"""
def main(save_dir):
    img_size = (320, 180)
    np.random.seed(100)
    driving_datas_categories = ['/media/hyunkun/ReT7/KATRI_BLOCKAGE_TEST/v4_not_rain',
                                 '/media/hyunkun/ReT7/KATRI_BLOCKAGE_TEST/v4_rain',
                                 '/media/hyunkun/ReT7/KATRI_BLOCKAGE_TEST/v5_not_rain',
                                 '/media/hyunkun/ReT7/KATRI_BLOCKAGE_TEST/v5_rain']
    cnt = 0

    for category in driving_datas_categories:
        for event in os.listdir(category):
            if not os.path.isdir(f'{category}/{event}'):
                # 백파일이 아니라 백파일을 디코딩한 이미지를 원함
                continue
            cameras = os.listdir(f'{category}/{event}')
            splits = ['train'] * 4 + ['val'] * 2 + ['test'] * 2
            random.shuffle(splits)
            for camera_num, camera in enumerate(cameras):
                split = splits[camera_num]
                img_dir = f'{category}/{event}/{camera}'
                img_names = os.listdir(img_dir)
                for img_name in img_names:
                    jpg_dst = f'{save_dir}/imgs/jpg_format/{split}/rf_{cnt}.jpg'
                    pt_dst = f'{save_dir}/imgs/pt_format/{split}/rf_{cnt}.pt'
                    cnt += 1
                    cvimg = cv2.imread(f'{img_dir}/{img_name}')

                    cvimg = cv2.resize(cvimg, img_size)
                    cv2.imwrite(jpg_dst, cvimg)
                    npimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
                    npimg = np.swapaxes(npimg, 0, 1)
                    npimg = np.swapaxes(npimg, 0, 2)

                    img = torch.from_numpy(npimg)
                    with open(f'{pt_dst}', 'wb') as f:
                        torch.save((img, torch.tensor(0)), f)


if __name__=='__main__':
    save_dir = '/media/hyunkun/ReT7/CameraBlockedDataset'
    main(save_dir)
