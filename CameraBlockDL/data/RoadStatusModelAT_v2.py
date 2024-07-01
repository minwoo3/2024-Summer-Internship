import os
import random
import cv2
import numpy as np
import torch
import argparse
import pickle
import csv
from tqdm import tqdm
def csvwriter(csv_dir, target_list):
        with open(csv_dir, 'w', newline="") as file:
            writer = csv.writer(file)
            writer.writerows(target_list)

def image_list_append(dataset_path, scene_list):
    img_list = []
    for scene in tqdm(scene_list):
        # print(scene)
        if 'NIA' in dataset_path:
            scene_abs_path = f'{dataset_path}/{scene}/image0/'
            try:
                scene_img_list = os.listdir(scene_abs_path)
                cnt = 0
                for img in scene_img_list:
                    if '@' not in img and cnt % 10 == 0:
                        img_path = scene_abs_path + img
                        # print(img_path)
                        img_list.append(img_path)
                    cnt += 1
            except FileNotFoundError:
                continue
        
        elif '벚꽃' in dataset_path:
            scene_abs_path = f'{dataset_path}/{scene}/camera_0/'
            try:
                scene_img_list = os.listdir(scene_abs_path)
                for img in scene_img_list:
                    if '@' not in img:
                        img_path = scene_abs_path + img
                        img_list.append(img_path)
            except FileNotFoundError:
                continue

    return img_list

def main(nia_img_dir, cbtree_img_dir ,clean_csv_save_dir, dirty_csv_save_dir):
    global args
    # clear
    # if args.clear:
    #     with open(clean_csv_save_dir,"w") as file:
    #         file.write("")
    #     with open(dirty_csv_save_dir,"w") as file:
    #         file.write("")

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
            cbtree_scenes.append(f'{day}/{scene}')

    nia_train_scenes = list(np.random.choice(nia_scenes, int(len(nia_scenes)*nia_train_ratio), replace = False))
    nia_rest_scenes = [x for x in nia_scenes if x not in nia_train_scenes]
    nia_val_scenes = list(np.random.choice(nia_rest_scenes, int(len(nia_scenes)*nia_val_ratio), replace = False))
    nia_test_scenes = [x for x in nia_rest_scenes if x not in nia_val_scenes]

    cbtree_train_scenes = list(np.random.choice(cbtree_scenes, int(len(cbtree_scenes)*cbtree_train_ratio), replace = False))
    cbtree_val_scenes = [x for x in cbtree_scenes if x not in cbtree_train_scenes]
    cbtree_test_scenes = cbtree_val_scenes
    
    print(f'{len(nia_scenes)} / {len(nia_train_scenes)} / {len(nia_val_scenes)} / {len(nia_test_scenes)}')
    print(f'{len(cbtree_scenes)} / {len(cbtree_train_scenes)} / {len(cbtree_val_scenes)} / {len(cbtree_test_scenes)}')
    
    clean_train_img = image_list_append(nia_img_dir, nia_train_scenes)
    clean_val_img = image_list_append(nia_img_dir, nia_val_scenes)
    clean_test_img = image_list_append(nia_img_dir, nia_test_scenes)
    
    dirty_train_img = image_list_append(cbtree_img_dir, cbtree_train_scenes)
    dirty_val_img = image_list_append(cbtree_img_dir, cbtree_val_scenes)
    dirty_test_img = image_list_append(cbtree_img_dir, cbtree_test_scenes)
    

    csvwriter(f'{clean_csv_save_dir}_train.csv', clean_train_img)
    csvwriter(f'{clean_csv_save_dir}_test.csv', clean_test_img)
    csvwriter(f'{clean_csv_save_dir}_val.csv', clean_val_img)

    csvwriter(f'{dirty_csv_save_dir}_train.csv', dirty_train_img)
    csvwriter(f'{dirty_csv_save_dir}_test.csv', dirty_test_img)
    csvwriter(f'{dirty_csv_save_dir}_val.csv', dirty_val_img)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # -c : clear 모드
    parser.add_argument('-c', '--clear', dest='clear', action = 'store_true')
    # parser.add_argument('-s', '--sample', dest='sample', action = 'store')
    args = parser.parse_args()
    # sampling_rate = int(args.sample)
    nia_img_dir = '/media/rideflux/T7/2024-Summer-Internship/NIA2021'
    cbtree_img_dir = '/media/rideflux/T7/2024-Summer-Internship/벚꽃'
    clean_csv_save_dir = '/media/rideflux/T7/2024-Summer-Internship/scene/clean'
    dirty_csv_save_dir = '/media/rideflux/T7/2024-Summer-Internship/scene/dirty'
    main(nia_img_dir, cbtree_img_dir, clean_csv_save_dir, dirty_csv_save_dir)