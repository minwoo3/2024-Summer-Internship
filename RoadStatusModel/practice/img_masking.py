import random
import json
import csv
import cv2
import getpass
import argparse
import numpy as np
import matplotlib.pyplot as plt
username = getpass.getuser()
csv_dir = 'sorted_data.csv'
nas_dir = '/home/rideflux/Public/LaneLineCamera/LaneLabel'
ssd_dir = f'/media/{username}/T7/2024-Summer-Internship/NIA2021'

class Viewer():
    def __init__(self, csv_path, index):
        self.csv_path = csv_path
        with open(csv_path,'r',newline='') as f:
            data = list(csv.reader(f))
        self.img_path, self.img_label= [], []
        # 30000,dirty
        for path, label in data:
            self.img_path.append(path)
            self.img_label.append(int(label))
        self.curr_i = index
        self.classes = ['clean','dirty']
        self.nas_dir = '/home/rideflux/Public/LaneLineCamera/lane_label_image0'
        self.ssd_dir = f'/media/{username}/T7/2024-Summer-Internship/NIA2021'

    def change_curr_dirs(self, dif):
        self.curr_i += dif
        if self.curr_i == len(self.img_path):
            print('end of list')
            self.curr_i -= dif
	
    def close(self, index):
        print(f'Work has been finished. Current Index : {index}')

    def view(self):
        while True:
            scene_num, _, img_num = self.img_path[self.curr_i].split('/')
            curr_img_path = f'{self.ssd_dir}/{self.img_path[self.curr_i]}'
            curr_bin_path = f'{self.nas_dir}/{scene_num}/{img_num[:10]}bin'
            curr_img_label = self.img_label[self.curr_i]
            img = cv2.imread(curr_img_path)
            img = cv2.resize(img,(1280,720))
            bin = np.fromfile(curr_bin_path, dtype = bool).reshape(930, 1440)
            uy, ux = bin.nonzero()
            ux = ux / 1440 * 1280
            uy = uy / 930 * 720
            
            ux, uy = np.clip(ux.astype(int), 0, 1280 - 1), np.clip(uy.astype(int), 0, 720 - 1)
            mask = np.zeros(img.shape[:2],dtype=np.uint8)
            # print(mask.shape)
            mask[uy, ux] = 255

            kernel_size = 50
            kernel = np.ones((kernel_size, kernel_size), np.uint8)

            # 차선 두께 증가
            mask = cv2.dilate(mask, kernel, iterations=1)
            extracted = cv2.bitwise_and(img,img,mask=mask)
            cv2.imshow('img',extracted)
            # img[uy, ux, 0] = 255
            # img[uy, ux, 1] = 0
            # img[uy, ux, 2] = 0

            # cv2.putText(img, f"{curr_img_path} {self.classes[curr_img_label]} {self.curr_i}/{len(self.img_path)}",(10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0), 2)
            # cv2.imshow('viewer', img)

            pressed = cv2.waitKeyEx(15)
            if pressed == 27: self.close(self.curr_i); break # Esc
            elif pressed == 96: self.change_curr_dirs(-1) # `
            elif pressed == 9: self.change_curr_dirs(1) # Tab
            elif pressed == 48: self.img_label[self.curr_i] = 0; self.change_curr_dirs(1)
            elif pressed == 49: self.img_label[self.curr_i] = 1; self.change_curr_dirs(1)
            
parser = argparse.ArgumentParser()
# parser.add_argument('--path', '-p', dest='path', required=True)
parser.add_argument('--index', '-i', dest='index',type = int, default = 0)
args = parser.parse_args()

viewer = Viewer(csv_dir,args.index)   
viewer.view()

# # plt.subplot(221)
# # plt.imshow(ptf_img, cmap='gray')

# # plt.subplot(222)
# # plt.imshow(mag_spectrum, cmap='gray')

# # plt.subplot(223)
# # plt.imshow(mag_spectrum_2, cmap='gray')


# # plt.subplot(224)
# # plt.imshow(inv_fft, cmap='gray')
# # plt.show()


# # cv2.imshow('crop',ptf_img)
# # # cv2.imshow('fft', inv_fft)

# cv2.waitKey(0)
# cv2.destroyWindow('img')
