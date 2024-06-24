import os
import pandas as pd
import cv2
import numpy as np
import torch
"""
camera blocked 데이터를 학습할 수 있는 형식으로 변환하고 annotation csv 파일을 만드는 파일
"""
def main(save_dir):
    cnt = 0
    img_size = (320, 180)
    data_dir = '/media/hyunkun/ReT7/KATRI_BLOCKAGE_TEST'
    train = ['2023-09-18-09-43-57_gv80_v4_1/camera_0',
             '2023-09-18-09-43-57_gv80_v4_1/camera_1']
    validation = ['2023-09-14-17-25-54_gv80_v4_1/camera_0',
                  '2023-09-14-17-25-54_gv80_v4_1/camera_1']
    test = ['2023-09-14-17-20-29_gv80_v4_1/camera_0',
                  '2023-09-14-17-20-29_gv80_v4_1/camera_1']
    splits = {'train': train,
             'val': validation,
             'test': test}
    for split, camera_dirs in splits.items():
        data_dict = {'img_file': [], 'label': []}
        for camera_dir in camera_dirs:
            blocked_dir = f'{data_dir}/{camera_dir}/blocked'
            normal_dir = f'{data_dir}/{camera_dir}/normal'
            for img_name in os.listdir(blocked_dir):
                jpg_dst = f'{save_dir}/imgs/jpg_format/{split}/blocked_{cnt}.jpg'
                pt_dst = f'{save_dir}/imgs/pt_format/{split}/blocked_{cnt}.pt'
                cnt += 1
                cvimg = cv2.imread(f'{data_dir}/{camera_dir}/blocked/{img_name}')

                cvimg = cv2.resize(cvimg, img_size)
                cv2.imwrite(jpg_dst, cvimg)
                npimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
                npimg = np.swapaxes(npimg, 0, 1)
                npimg = np.swapaxes(npimg, 0, 2)

                img = torch.from_numpy(npimg)
                with open(f'{pt_dst}', 'wb') as f:
                    torch.save((img, torch.tensor(1)), f)



if __name__=='__main__':
    save_dir = '/media/hyunkun/ReT7/CameraBlockedDataset'
    main(save_dir)
