import os
import pandas as pd
import cv2
import numpy as np
import torch
import argparse
import random
"""
camera clean 데이터를 학습할 수 있는 형식으로 변환하고 annotation csv 파일을 만드는 파일
"""
def main(data_dir,save_dir, sample_num,total_ratio):
    with open(f'/home/rideflux/2024-Summer-Internship/CameraBlockDL/data/NIA2021.txt') as f:
        nia_total_scenes = f.read().splitlines()
    sampled_scenes = []
    for ratio in total_ratio:
        sampled_scenes.append(list(np.random.choice(nia_total_scenes,size = int(sample_num*ratio), replace = False)))

    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # -c : clear 모드
    parser.add_argument('-c', '--clear', dest='clear', action = 'store_true')
    args = parser.parse_args()

    data_dir = '/media/rideflux/T7/2024-Summer-Internship/NIA2021'
    save_dir = '/media/rideflux/T7/RoadDirtDataset'
    main(data_dir,save_dir,sample_num = 100, total_ratio=[0.7,0.15,0.15])
