import os
import numpy as np
import argparse
import csv
from tqdm import tqdm
import getpass

def csvwriter(csv_dir, target_list):
    with open(csv_dir, 'w', newline="") as file:
        writer = csv.writer(file)
        writer.writerows(target_list)

def imgpath(dataset_path, scene_list):
    img_list = []
    for scene in tqdm(scene_list):
        if 'NIA' in dataset_path:
            scene_abs_path = f'{dataset_path}/{scene}/image0/'
            try:
                scene_img_list = os.listdir(scene_abs_path)
                cnt = 0
                for img in scene_img_list:
                    if '@' not in img and cnt % 10 == 0:
                        img_path = scene_abs_path + img
                        img_list.append([img_path,0])
                    cnt += 1
            except FileNotFoundError:
                continue
        
        elif '벚꽃' in dataset_path or 'Raw' in dataset_path:
            scene_abs_path = f'{dataset_path}/{scene}/camera_0/'
            try:
                scene_img_list = os.listdir(scene_abs_path)
                for img in scene_img_list:
                    if '@' not in img:
                        img_path = scene_abs_path + img
                        img_list.append([img_path,1])
            except FileNotFoundError:
                continue

    return img_list

def datasplit(total, train_ratio, val_ratio, test_ratio):
    train_scenes = list(np.random.choice(total, int(len(total)*train_ratio), replace = False))
    rest_scenes = [x for x in total if x not in train_scenes]
    val_scenes = list(np.random.choice(rest_scenes, int(len(total)*val_ratio), replace = False))
    test_scenes = [x for x in rest_scenes if x not in val_scenes]
    return train_scenes, val_scenes, test_scenes

def annotate(nia_img_dir, cbtree_img_dir, nas_img_dir, nas_dirty_csv_dir, nas_clean_csv_dir ,csv_save_dir):
    global args

    nia_train_scenes, nia_val_scenes, nia_test_scenes = [], [], []
    cbtree_train_scenes, cbtree_val_scenes, cbtree_test_scenes = [], [], []
    nas_train_scenes,nas_val_scenes, nas_test_scenes = [], [], []

    nia_train_ratio, nia_val_ratio, nia_test_ratio = 0.7, 0.15, 0.15
    cbtree_train_ratio, cbtree_val_ratio, cbtree_test_ratio = 0.5, 0.25, 0.25
    nas_train_ratio, nas_val_ratio, nas_test_ratio = 0.625, 0.1875, 0.1875
    
    ### [NIA2021] Scene 리스트
    nia_scenes = os.listdir(nia_img_dir)
    ### [벚꽃] Scene 리스트 
    days = os.listdir(cbtree_img_dir)
    cbtree_scenes = []
    for day in days:
        scenes = os.listdir(f'{cbtree_img_dir}/{day}')
        for scene in scenes:
            cbtree_scenes.append(f'{day}/{scene}')
    
    nas_dirty_scenes, nas_clean_scenes = [], []
    ### [Nas] Scene 리스트 : 이미 정리된 CSV 읽어오기
    with open(nas_clean_csv_dir,'r',newline='') as f:
        data = list(csv.reader(f))
    for scene in data:
        nas_clean_scenes.append(scene[0])
    
    with open(nas_dirty_csv_dir,'r',newline='') as f:
        data = list(csv.reader(f))
    for scene in data:
        nas_dirty_scenes.append(scene[0])

    nia_train_scenes, nia_val_scenes, nia_test_scenes = datasplit(nia_scenes, nia_train_ratio, nia_val_ratio, nia_test_ratio)
    cbtree_train_scenes, cbtree_val_scenes, cbtree_test_scenes = datasplit(cbtree_scenes, cbtree_train_ratio, cbtree_val_ratio, cbtree_test_ratio)


    nas_dirty_scenes_reduced = list(np.random.choice(nas_dirty_scenes, 160, replace=False))
    nas_dirty_train_scenes, nas_dirty_val_scenes, nas_dirty_test_scenes = datasplit(nas_dirty_scenes_reduced, nas_train_ratio, nas_val_ratio, nas_test_ratio)
    # print(f'{len(nia_scenes)} / {len(nia_train_scenes)} / {len(nia_val_scenes)} / {len(nia_test_scenes)}')
    # print(f'{len(cbtree_scenes)} / {len(cbtree_train_scenes)} / {len(cbtree_val_scenes)} / {len(cbtree_test_scenes)}')
    # print(f'{len(nas_dirty_scenes)} / {len(nas_dirty_train_scenes)} / {len(nas_dirty_val_scenes)} / {len(nas_dirty_test_scenes)}')
    
    nia_train_img = imgpath(nia_img_dir, nia_train_scenes)
    nia_val_img = imgpath(nia_img_dir, nia_val_scenes)
    nia_test_img = imgpath(nia_img_dir, nia_test_scenes)
    
    cbtree_train_img = imgpath(cbtree_img_dir, cbtree_train_scenes)
    cbtree_val_img = imgpath(cbtree_img_dir, cbtree_val_scenes)
    cbtree_test_img = imgpath(cbtree_img_dir, cbtree_test_scenes)
    
    nas_dirty_train_img = imgpath(nas_img_dir, nas_dirty_train_scenes)
    nas_dirty_val_img = imgpath(nas_img_dir, nas_dirty_val_scenes)
    nas_dirty_test_img = imgpath(nas_img_dir, nas_dirty_test_scenes)

    train_img = nia_train_img + cbtree_train_img + nas_dirty_train_img
    val_img = nia_val_img + cbtree_val_img + nas_dirty_val_img
    test_img = nia_test_img + cbtree_test_img + nas_dirty_test_img

    csvwriter(f'{csv_save_dir}/train.csv', train_img)
    csvwriter(f'{csv_save_dir}/test.csv', test_img)
    csvwriter(f'{csv_save_dir}/val.csv', val_img)

    # csvwriter(f'{dirty_csv_save_dir}_train.csv', dirty_train_img)
    # csvwriter(f'{dirty_csv_save_dir}_test.csv', dirty_test_img)
    # csvwriter(f'{dirty_csv_save_dir}_val.csv', dirty_val_img)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # -c : clear 모드
    parser.add_argument('-c', '--clear', dest='clear', action = 'store_true')
    args = parser.parse_args()

    username = getpass.getuser()

    nia_img_dir = f'/media/{username}/T7/2024-Summer-Internship/NIA2021'
    cbtree_img_dir = f'/media/{username}/T7/2024-Summer-Internship/벚꽃'
    nas_img_dir = f'/home/{username}/GeneralCase/Raw'
    nas_dirty_csv_dir = f'/media/{username}/T7/2024-Summer-Internship/scene/30000_dirty.csv'
    nas_clean_csv_dir = f'/media/{username}/T7/2024-Summer-Internship/scene/30000_clean.csv'

    csv_save_dir = f'/media/{username}/T7/2024-Summer-Internship/scene'

    annotate(nia_img_dir, cbtree_img_dir, nas_img_dir, nas_dirty_csv_dir, nas_clean_csv_dir ,csv_save_dir)

