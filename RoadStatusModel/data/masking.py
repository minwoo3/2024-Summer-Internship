import cv2, os, re, time, csv, argparse, getpass
import numpy as np

def csvwriter(csv_dir, target_list):
    with open(csv_dir, 'w', newline="") as file:
        writer = csv.writer(file)
        writer.writerows(target_list)

class Viewer():
    def __init__(self, csv_path, index):
        self.csv_path = csv_path
        self.username = getpass.getuser()
        with open(csv_path,'r',newline='') as f:
            data = list(csv.reader(f))
        self.img_path, self.img_label= [], []
        self.sata_dir = f'/media/{self.username}/sata-ssd'
        self.t7_dir = f'/media/{self.username}/T7/2024-Summer-Internship'
        for path, label in data:
            self.img_path.append(path)
            self.img_label.append(int(label))
        self.curr_i = index
        self.classes = ['clean','dirty']
        self.img_width, self.img_height = 1280, 720
        self.mask_width, self.mask_height = 1280, 720
        self.mask, self.kernel_size = 2, 50

    def change_curr_dirs(self, dif):
        self.curr_i += dif
        if self.curr_i == len(self.img_path):
            print('end of list')
            self.curr_i -= dif
        elif self.curr_i == 0:
            print('first of list')
            self.curr_i += dif
	
    def save(self, index):
        result = []                                                             
        for i in range(len(self.img_path)):
            result.append([self.img_path[i],self.img_label[i]])
        csvwriter(self.csv_path,result)
        print(f'Work has been saved. Current Index : {index}')

    def openbin(self,dir):
        if '연석' in dir:
            if 'NIA' in dir:
                bin = np.fromfile(dir.replace('image0/',''),dtype = int).reshape(-1,2)
            elif '벚꽃' in dir or 'GeneralCase' in dir:
                bin = np.fromfile(dir, dtype = np.float16).reshape(-1,3)
            uy, ux = bin[:,1], bin[:,0]
        elif '차선' in dir:
            if 'NIA' in dir:
                bin = np.fromfile(dir.replace('image0/',''), dtype = bool).reshape(930, 1440)
                uy, ux = bin.nonzero()
            elif '벚꽃' in dir or 'GeneralCase' in dir:
                bin = np.fromfile(dir, dtype = np.float16).reshape(-1,3)
                uy, ux = bin[:,1], bin[:,0]
        ux, uy = ux / 1440 * self.mask_width, uy / 930 * self.mask_height
        ux, uy = np.clip(ux.astype(int), 0, self.mask_width - 1), np.clip(uy.astype(int), 0, self.mask_height - 1)
        return ux, uy
    
    def makemask(self, mode):
        curb_x, curb_y = self.openbin(self.curr_curb)
        lane_x, lane_y = self.openbin(self.curr_lane)
        if mode == 2:
            mask = np.zeros((self.mask_height,self.mask_width),dtype=np.uint8)
            mask[curb_y, curb_x] = 255
            mask[lane_y, lane_x] = 255
        elif mode == 1:
            mask = np.zeros((self.mask_height,self.mask_width),dtype=np.uint8)
            mask[lane_y, lane_x] = 255
        elif mode == 0:
            mask = np.zeros((self.mask_height,self.mask_width),dtype=np.uint8)
            mask[curb_y, curb_x] = 255

        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        return mask, curb_x, curb_y, lane_x, lane_y

    def getmask(self):
        if self.mask == 2:
            mask, curb_x, curb_y, lane_x, lane_y = self.makemask(2)
        elif self.mask == 1:
            mask, curb_x, curb_y, lane_x, lane_y = self.makemask(1)
        elif self.mask == 0:
            mask, curb_x, curb_y, lane_x, lane_y = self.makemask(0)

        return mask, curb_x, curb_y, lane_x, lane_y

    def changemask(self):
        self.mask += 1
        self.mask = self.mask % 3

    def changekernelsz(self,diff):
        self.kernel_size += diff

    def view(self):
        while True:
            #/NIA2021/10004/image0/10004_040.jpg
            if 'NIA' in self.img_path[self.curr_i] or '벚꽃' in self.img_path[self.curr_i]:
                curr_img_path = self.t7_dir + self.img_path[self.curr_i]
            elif 'GeneralCase' in self.img_path[self.curr_i]:
                curr_img_path = self.sata_dir + self.img_path[self.curr_i]
            self.curr_curb = self.sata_dir + '/camera_inference/연석' + self.img_path[self.curr_i][:-4] + '.bin'
            self.curr_lane = self.sata_dir + '/camera_inference/차선' + self.img_path[self.curr_i][:-4] + '.bin'
            curr_img_label = self.img_label[self.curr_i]
            try:
                img = cv2.imread(curr_img_path)
                img = cv2.resize(img,(1280, 720))
                mask, curb_x, curb_y, lane_x, lane_y = self.getmask()
                extracted_img = cv2.bitwise_and(img,img,mask = mask)

                for x, y in zip(lane_x, lane_y):
                    cv2.circle(extracted_img,(int(x),int(y)),3,(255,0,0),thickness= -1)
                for x, y in zip(curb_x, curb_y):
                    cv2.circle(extracted_img,(int(x),int(y)),3,(0,255,0),thickness= -1)
                    
                cv2.putText(extracted_img, f"{self.img_path[self.curr_i]} {self.classes[curr_img_label]} {self.curr_i}/{len(self.img_path)} kernel: {self.kernel_size}",(10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 2)
                cv2.imshow('img',extracted_img)

            except cv2.error as e:
                print(f'{curr_img_path} is not available. Current index : {self.curr_i}')
                self.save(self.curr_i -1)
                self.change_curr_dirs(1)

            pressed = cv2.waitKeyEx(15)
            if pressed == 27: self.save(self.curr_i); break # Esc
            elif pressed == 56 or pressed == ord('w'): self.change_curr_dirs(100) # 8
            elif pressed == 54 or pressed == ord('d'): self.change_curr_dirs(1) # 6
            elif pressed == 52 or pressed == ord('a'): self.change_curr_dirs(-1) # 4
            elif pressed == 50 or pressed == ord('s'): self.change_curr_dirs(-100) # 2
            elif pressed == 48: self.img_label[self.curr_i] = 0; self.change_curr_dirs(1)
            elif pressed == 49: self.img_label[self.curr_i] = 1; self.change_curr_dirs(1)
            elif pressed == ord('o'): self.changemask()
            elif pressed == ord('m'): self.changekernelsz(1)
            elif pressed == ord('n'): self.changekernelsz(-1)

            
parser = argparse.ArgumentParser()
parser.add_argument('--path', '-p', dest='path', required=True)
parser.add_argument('--index', '-i', dest='index',type = int, default = 0)
args = parser.parse_args()

viewer = Viewer(args.path,args.index)   
viewer.view()
