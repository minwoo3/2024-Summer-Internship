import os
import random
import cv2
import numpy as np
import torch
import argparse
import pickle
import csv

def main(clean_data_dir, dirty_data_dir ,clean_csv_save_dir, dirty_csv_save_dir,p_save_dir,sampling_rate):
    global args
    # clear
    if args.clear:
        with open(clean_csv_save_dir,"w") as file:
            file.write("")
        open(p_save_dir, 'w').close()
        with open(dirty_csv_save_dir,"w") as file:
            file.write("")
        open(p_save_dir, 'w').close()

    clean, dirty = [], []
    ### NIA2021 이미지 리스트
    scenes = os.listdir(clean_data_dir)
    for scene in scenes:
        print(f'NIA2021 Scene : {scene}')
        try:
            images = os.listdir(f'{clean_data_dir}/{scene}/image0')
            ### frame 1/10로 sampling
            cnt = 0
            for image in images:
                if '@' not in image and (cnt % sampling_rate == 0):
                    clean.append([f'{clean_data_dir}/{scene}/image0/{image}',0])
                    # clean_p[f'{clean_data_dir}/{scene}/image0/{image}'] = 0
                cnt += 1
        except:
            print('no image0')
            continue
    ### 벚꽃 이미지 리스트 
    days = os.listdir(dirty_data_dir)
    for day in days:
        scenes = os.listdir(f'{dirty_data_dir}/{day}')
        for scene in scenes:
            print(f'벚꽃 {day} day / {scene} scene')
            try:
                images = os.listdir(f'{dirty_data_dir}/{day}/{scene}/camera_0')
                for image in images:
                    dirty.append([f'{dirty_data_dir}/{day}/{scene}/camera_0/{image}', 1])
                    # dirty_p[f'{dirty_data_dir}/{day}/{scene}/camera_0/{image}'] = 1
            except:
                print('no camera_0')
                continue

    # data = clean + dirty
    with open(clean_csv_save_dir,'w',newline="") as file:
        writer = csv.writer(file)
        writer.writerows(clean)
    
    with open(dirty_csv_save_dir,'w',newline="") as file:
        writer = csv.writer(file)
        writer.writerows(dirty)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # -c : clear 모드
    parser.add_argument('-c', '--clear', dest='clear', action = 'store_true')
    parser.add_argument('-s', '--sample', dest='sample', action = 'store')
    args = parser.parse_args()
    sampling_rate = int(args.sample)
    clean_data_dir = '/media/rideflux/T7/2024-Summer-Internship/NIA2021'
    dirty_data_dir = '/media/rideflux/T7/2024-Summer-Internship/벚꽃'
    clean_csv_save_dir = '/home/rideflux/2024-Summer-Internship/CameraBlockDL/data/clean.csv'
    dirty_csv_save_dir = '/home/rideflux/2024-Summer-Internship/CameraBlockDL/data/dirty.csv'
    p_save_dir = '/home/rideflux/2024-Summer-Internship/CameraBlockDL/data/total.p'
    main(clean_data_dir, dirty_data_dir, clean_csv_save_dir, dirty_csv_save_dir, p_save_dir, sampling_rate)