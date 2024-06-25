import os
import random
import cv2
import numpy as np
import torch
import argparse
def main(data_dir,save_dir):
    global args
    file_path = "/home/rideflux/2024-Summer-Internship/CameraBlockDL/data/NIA2021.txt"
    if args.clear:
        with open(file_path,"w") as file:
            file.write("")
    
    list_tot = []
    scenes = os.listdir(data_dir)
    # print(scene)
    for scene in scenes:
        print(f'scene : {scene}')
        try:
            images = os.listdir(f'{data_dir}/{scene}/image0')
            for image in images:
                # print(image)
                if '@' not in image:
                    list_tot.append(f'{data_dir}/{scene}/image0/{image}')
        except:
            print('no image0')
            continue

    with open(file_path, "w") as file:
        for item in list_tot:
            file.write(item + "\n")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # -c : clear 모드
    parser.add_argument('-c', '--clear', dest='clear', action = 'store_true')
    args = parser.parse_args()
    
    data_dir = '/media/rideflux/T7/2024-Summer-Internship/NIA2021'
    save_dir = '/media/rideflux/T7/RoadDirtDataset'
    main(data_dir,save_dir)