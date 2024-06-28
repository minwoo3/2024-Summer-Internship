import os
import random
import cv2
import numpy as np
import torch
import argparse
import pickle
import csv

def main(nia_img_dir, cbtree_img_dir ,clean_csv_save_dir, dirty_csv_save_dir):
    global args
    # clear
    if args.clear:
        with open(clean_csv_save_dir,"w") as file:
            file.write("")
        with open(dirty_csv_save_dir,"w") as file:
            file.write("")

    clean, dirty = [], []
    nia_scenes, cbtree_scenes = [], []
    nia_train_scenes, nia_val_scenes, nia_test_scenes = [], [], []
    cbtree_train_scenes, cbtree_val_scenes, cbtree_test_scenes = [], [], []
    nia_train_ratio, nia_val_ratio, nia_test_ratio = 0.7, 0.15, 0.15
    cbtree_train_ratio, cbtree_val_ratio, cbtree_test_ratio = 0.5, 0.5, 0.5
    ### NIA2021 이미지 리스트
    # scenes = os.listdir(nia_img_dir)
    # nia_scenes = []
    # for scene in scenes:
    #     nia_scenes.append([scene])
    nia_scenes = os.listdir(nia_img_dir)
    # ratio 기준 scene 단위 split

    ### 벚꽃 이미지 리스트 
    days = os.listdir(cbtree_img_dir)
    cbtree_scenes = []
    for day in days:
        scenes = os.listdir(f'{cbtree_img_dir}/{day}')
        for scene in scenes:
            cbtree_scenes.append([f'{day}/{scene}'])

    nia_train_scenes = list(np.random.choice(nia_scenes, int(len(nia_scenes)*nia_train_ratio)))
    nia_rest_scenes = [x for x in nia_scenes if x not in nia_train_scenes]
    nia_val_scenes = list(np.random.choice(nia_rest_scenes, int(len(nia_scenes)*nia_val_ratio)))
    nia_test_scenes = [x for x in nia_rest_scenes if x not in nia_val_scenes]
    print(len(nia_scenes))
    # print(len(nia_train_scenes))
    print(len(nia_rest_scenes))
    print(len(nia_test_scenes))
    # with open(clean_csv_save_dir,'w',newline="") as file:
    #     writer = csv.writer(file)
    #     writer.writerows(nia_scenes)
    
    # with open(dirty_csv_save_dir,'w',newline="") as file:
    #     writer = csv.writer(file)
    #     writer.writerows(cbtree_scenes)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # -c : clear 모드
    parser.add_argument('-c', '--clear', dest='clear', action = 'store_true')
    # parser.add_argument('-s', '--sample', dest='sample', action = 'store')
    args = parser.parse_args()
    # sampling_rate = int(args.sample)
    nia_img_dir = '/media/rideflux/T7/2024-Summer-Internship/NIA2021'
    cbtree_img_dir = '/media/rideflux/T7/2024-Summer-Internship/벚꽃'
    clean_csv_save_dir = '/home/rideflux/2024-Summer-Internship/CameraBlockDL/data/clean_train.csv'
    dirty_csv_save_dir = '/home/rideflux/2024-Summer-Internship/CameraBlockDL/data/dirty_train.csv'
    main(nia_img_dir, cbtree_img_dir, clean_csv_save_dir, dirty_csv_save_dir)