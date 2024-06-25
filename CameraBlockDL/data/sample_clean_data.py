import os
import pandas as pd
import cv2
import numpy as np
import torch
import argparse
"""
camera clean 데이터를 학습할 수 있는 형식으로 변환하고 annotation csv 파일을 만드는 파일
"""
def main(save_dir):
    global args
    cnt = 0
    img_size = (320, 180)
    data_dir = '/media/rideflux/T7/2024-Summer-Internship/NIA2021'
    
    train_dir, valid_dir, test_dir = '10002', '10003', '10004'

    train = [f'{train_dir}/image0', f'{train_dir}/image1']
    validation = [f'{valid_dir}/image0', f'{valid_dir}/image1']
    test = [f'{test_dir}/image0', f'{test_dir}/image1']
    
    splits = {'train': train,
             'val': validation,
             'test': test}
    
    for split, camera_dirs in splits.items():
        
        for camera_dir in camera_dirs:
            clean_dir = f'{data_dir}/{camera_dir}'
            
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

            for img_name in os.listdir(clean_dir):
                jpg_dst = f'{save_dir}/imgs/jpg_format/{split}/clean_{cnt}.jpg'
                pt_dst = f'{save_dir}/imgs/pt_format/{split}/clean_{cnt}.pt'
                cnt += 1
                cvimg = cv2.imread(f'{data_dir}/{camera_dir}/{img_name}')
                # print(f'{data_dir}/{camera_dir}/{img_name}')
                
                # cvimg = cv2.resize(cvimg, img_size)
                # cv2.imwrite(jpg_dst, cvimg)
                # npimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
                # npimg = np.swapaxes(npimg, 0, 1)
                # npimg = np.swapaxes(npimg, 0, 2)
                # img = torch.from_numpy(npimg)
                # with open(f'{pt_dst}', 'wb') as f:
                #     torch.save((img, torch.tensor(1)), f)

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
