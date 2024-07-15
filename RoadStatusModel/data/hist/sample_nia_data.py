import os
import random
import cv2
import numpy as np
import torch
dir_path = os.path.dirname(os.path.realpath(__file__))

"""
니아 데이터에서 블락 학습에 사용할 주간 야간 데이터를 샘플하고 320x180 사이즈로 리사이즈 해서 pt 파일로 저장한다.
니아 데이터 중에는 80개의 야간 데이터가 있고 주간 데이터는 20개를 랜덤으로 선별 한다.
야간 데이터는 메타데이터 상에서 코드 17 이다.
"""
def main(nia_data_dir, save_dir):
    img_size = (320, 180)
    with open(f'{dir_path}/datas/nia_night_scenes.txt', 'r') as f:
        nia_night_scenes = f.read().splitlines()

    all_scenes = os.listdir(nia_data_dir)
    nia_day_scenes = [scene for scene in all_scenes if scene not in nia_night_scenes]

    np.random.seed(100)
    sampled_scenes = list(np.random.choice(nia_day_scenes, replace=False, size=20)) + nia_night_scenes
    split_scenes = ['train'] * 70 + ['val'] * 15 + ['test'] * 15
    random.shuffle(split_scenes)
    cnt = 0
    for i, scene in enumerate(sampled_scenes):
        split = split_scenes[i]
        for camera_num in range(5):
            img_dir = f'{nia_data_dir}/{scene}/image{camera_num}'
            img_names = os.listdir(img_dir)
            for img_name in img_names:
                jpg_dst = f'{save_dir}/imgs/jpg_format/{split}/nia_{cnt}.jpg'
                pt_dst = f'{save_dir}/imgs/pt_format/{split}/nia_{cnt}.pt'
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
    nia_data_dir = '/media/hyunkun/nia-t7/NIA_2021_FULL_RAW'
    save_dir = '/media/hyunkun/ReT7/CameraBlockedDataset'
    main(nia_data_dir, save_dir)
