import cv2
import os
import argparse
import re
import numpy as np
import csv
import time

def csvwriter(csv_dir, target_list):
    with open(csv_dir, 'w', newline="") as file:
        writer = csv.writer(file)
        writer.writerows(target_list)

def ptf(pts,img):
    tl, tr, bl, br = pts
    pts1 = np.float32([tl, tr, bl, br])

    w1 = abs(br[0]-bl[0])
    w2 = abs(tr[0]-tl[0])
    width = int(max([w1,w2]))
    h1 = abs(br[1]-tr[1])
    h2 = abs(bl[1]-tl[1])
    height = int(max([h1,h2]))

    pts2 = np.float32([[0,0],[width-1,0],[0, height-1],[width-1,height-1]])

    transform_mat = cv2.getPerspectiveTransform(pts1,pts2)

    result = cv2.warpPerspective(img, transform_mat, (width, height))
    return result, transform_mat

def fft(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fft = np.fft.fft2(img)
    fshift = np.fft.fftshift(fft)
    mag_spectrum = 20*np.log(np.abs(fshift))
    (w,h) = fshift.shape
    cx,cy = int(w/2),int(h/2)
    n = 50
    fshift[cx-n:cx+n+1,cy-n:cy+n+1] = 0
    mag_spectrum_2 = 20*np.log(np.abs(0.01+fshift)).astype(int)
    fshift_i = np.fft.ifftshift(fshift)
    inv_fft = np.fft.ifft2(fshift_i).real
    inv_fft = inv_fft.astype(np.float32)/np.max(inv_fft)
    _, inv_fft = cv2.threshold(inv_fft, 0.3, 1.0, cv2.THRESH_BINARY)
    inv_fft[inv_fft<=0] = 0
    inv_fft[inv_fft > 0 ] = 255
    return inv_fft


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
        self.classes = ['not right','right']

    def change_curr_dirs(self, dif):
        self.curr_i += dif
        if self.curr_i == len(self.img_path):
            print('end of list')
            self.curr_i -= dif
	
    def save(self, index):
        result = []                                                             
        for i in range(len(self.img_path)):
            result.append([self.img_path[i],self.img_label[i]])
        csvwriter(self.csv_path,result)
        print(f'Work has been saved. Current Index : {index}')

    def view(self):
        while True:
            curr_img_path = self.img_path[self.curr_i]
            curr_img_label = self.img_label[self.curr_i]
            try:
                img = cv2.imread(curr_img_path)
                img = cv2.resize(img,(1280, 720))
                cv2.putText(img, f"{curr_img_path} {self.classes[curr_img_label]} {self.curr_i}/{len(self.img_path)}",(10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255), 2)
                cv2.imshow('img',img)
            except cv2.error as e:
                print(f'{curr_img_path} is not available. Current index : {self.curr_i}')
                self.save(self.curr_i -1)
                self.change_curr_dirs(1)

            
            # pts1 = [[350,400],[930,400],[5,600],[1275,600]]
            # ptf_img, transform_mat = ptf(pts1,img)
            # inverse_mat = np.linalg.inv(transform_mat)
            # original_image = cv2.warpPerspective(ptf_img, inverse_mat, (1280,720))
            # print(ptf_img.shape)
            # img = cv2.Canny(ptf_img,50,150)

            # fft_img = fft(img)
            
            # for x, y in pts1:
            #     cv2.circle(img, (x, y), 10, (0, 255, 0), -1)
            
            # cv2.imshow('viewer', fft_img)
            # cv2.imshow('ptf',ptf_img)
            # cv2.imshow('inverse',original_image)


            pressed = cv2.waitKeyEx(15)
            if pressed == 27: self.save(self.curr_i); break # Esc
            elif pressed == 96: self.change_curr_dirs(-1) # `
            elif pressed == 9: self.change_curr_dirs(1) # Tab
            elif pressed == 48: self.img_label[self.curr_i] = 0; self.change_curr_dirs(1)
            elif pressed == 49: self.img_label[self.curr_i] = 1; self.change_curr_dirs(1)
            
parser = argparse.ArgumentParser()
parser.add_argument('--path', '-p', dest='path', required=True)
parser.add_argument('--index', '-i', dest='index',type = int, default = 0)
args = parser.parse_args()

viewer = Viewer(args.path,args.index)   
viewer.view()
