import os
import pandas as pd
import cv2
import numpy as np
import torch
import argparse
"""
camera dirty 데이터를 학습할 수 있는 형식으로 변환하고 annotation csv 파일을 만드는 파일
"""
def main(save_dir):
    global args
    cnt = 0
    img_size = (320, 180)
    data_dir = '/media/rideflux/T7/2024-Summer-Internship/벚꽃/04-03'
    
    train_dir, valid_dir, test_dir = '2024-04-03-10-45-13_solati_v5_6_1837-1858', '2024-04-03-13-07-59_solati_v5_6_646-667', '2024-04-03-15-19-18_solati_v5_5_3-24'
    
    train = [f'{train_dir}/camera_0', f'{train_dir}/camera_1']
    validation = [f'{valid_dir}/camera_0', f'{valid_dir}/camera_1']
    test = [f'{test_dir}/camera_0', f'{test_dir}/camera_1']
    
    splits = {'train': train,
             'val': validation,
             'test': test}
    
    for split, camera_dirs in splits.items():

        for camera_dir in camera_dirs:
            dirty_dir = f'{data_dir}/{camera_dir}'

            jpg_folder_dir = f'{save_dir}/imgs/jpg_format/{split}'
            pt_folder_dir = f'{save_dir}/imgs/pt_format/{split}'
            # clear 모드의 경우, 폴더 내부 전체 삭제 후, 새로 불러오기
            if args.clear:
                if os.path.exists(jpg_folder_dir):
                    for file in os.scandir(jpg_folder_dir):
                        os.remove(file.path)
                    print(f'Cleared {jpg_folder_dir}')
                
                if os.path.exists(pt_folder_dir):
                    for file in os.scandir(pt_folder_dir):
                        os.remove(file.path)
                    print(f'Cleared {pt_folder_dir}')
                

            ### os.listdir(): 지정한 디렉토리 내의 모든 파일과 디렉토리의 리스트를 리턴
            for img_name in os.listdir(dirty_dir):
                jpg_dst = f'{save_dir}/imgs/jpg_format/{split}/dirty_{cnt}.jpg'
                pt_dst = f'{save_dir}/imgs/pt_format/{split}/dirty_{cnt}.pt'
                cnt += 1
                cvimg = cv2.imread(f'{data_dir}/{camera_dir}/{img_name}')
                try:
                    cvimg = cv2.resize(cvimg, img_size)
                    cv2.imwrite(jpg_dst, cvimg)
                    npimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
                    npimg = np.swapaxes(npimg, 0, 1)
                    npimg = np.swapaxes(npimg, 0, 2)
                    img = torch.from_numpy(npimg)
                    with open(f'{pt_dst}', 'wb') as f:
                        torch.save((img, torch.tensor(1)), f)
                except:
                    print('no image')
                    continue



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # -c : clear 모드
    parser.add_argument('-c', '--clear', dest='clear', action = 'store_true')
    args = parser.parse_args()

    save_dir = '/media/rideflux/T7/RoadDirtDataset'
    main(save_dir)
