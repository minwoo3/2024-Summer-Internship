import cv2, sys, os, argparse, re, getpass, csv, time, random
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from data.dataset import RoadStatusDataset
class Viewer():
    def __init__(self, csv_path, index):
        
        self.curr_i = index
        self.classes = ['clean','dirty']
        username = getpass.getuser()
        self.t7_dir = f'/media/{username}/T7/2024-Summer-Internship'
        self.sata_dir = f'/media/{username}/sata-ssd'
        self.kernel_size = 5
        self.mask, self.fft_flag, self.median_flag = False, True, False
        self.dataset = RoadStatusDataset(annotation_file= args.path, transform_flag= 'mask')
        example_img, _, _ = self.dataset[0]
        self.img_height, self.img_width = example_img.shape[-2:]  # (height, width)
    
    def change_curr_dirs(self, dif):
        self.curr_i += dif
        if self.curr_i == len(self.dataset):
            print('end of list')
            self.curr_i -= dif
	
    def change_kernel_size(self, dif):
        self.kernel_size += dif

    def applymask(self):
        if self.mask == False: self.mask = True
        else: self.mask = False
    
    def applyfft(self):
        if self.fft_flag == False: self.fft_flag = True
        else: self.fft_flag = False

    def applymedian(self):
        if self.median_flag == False: self.median_flag = True
        else: self.median_flag = False

    def fft(self,img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fft = np.fft.fft2(img)
        fshift = np.fft.fftshift(fft)
        mag_spectrum = 20*np.log(np.abs(fshift))
        (w,h) = fshift.shape
        cx,cy = int(w/2),int(h/2)
        n = self.kernel_size
        fshift[cx-n:cx+n+1,cy-n:cy+n+1] = 0
        mag_spectrum_2 = 20*np.log(np.abs(0.01+fshift)).astype(int)
        fshift_i = np.fft.ifftshift(fshift)
        inv_fft = np.fft.ifft2(fshift_i).real
        inv_fft = inv_fft.astype(np.float32)/np.max(inv_fft)
        _, inv_fft = cv2.threshold(inv_fft, 0.3, 1.0, cv2.THRESH_BINARY)
        inv_fft[inv_fft<=0] = 0
        inv_fft[inv_fft > 0 ] = 255
        return inv_fft

    def view(self):
        while True:
            curr_img, curr_label, curr_path = self.dataset[self.curr_i]
            curr_mask = np.array(curr_img[1,:,:]).astype(np.uint8)
            
            if 'NIA' in curr_path or '벚꽃' in curr_path:
                img = cv2.imread(self.t7_dir + curr_path)
            elif 'GeneralCase' in curr_path:
                img = cv2.imread(self.sata_dir + curr_path)
            img = cv2.resize(img, (self.img_width, self.img_height))
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            img[:self.img_height//2,:] = 0
    
            if self.fft_flag == True:
                img = self.fft(img)
            
            if self.median_flag == True:
                img = cv2.medianBlur(img,3,dst = None)

            if self.mask == True:
                img = img.astype(np.uint8)
                img = cv2.bitwise_and(img, img, mask=curr_mask)
                img = img.astype(np.float32) / 255

            
                # contours, hierachy = cv2.findContours(img.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
                # dst = np.zeros((self.img_height, self.img_width, 3), np.uint8)
                # for i in range(len(contours)):
                #     epsilon = 0.03 * cv2.arcLength(contours[i], closed = True)
                #     approx = cv2.approxPolyDP(contours[i], epsilon, closed =True)
                #     cv2.drawContours(dst, [approx], -1, (0,255,0), 2)
                    
                    # cv2.drawContours(dst,contours, i, (255,255,255), 1, cv2.LINE_AA)
                    # x, y, w, h = cv2.boundingRect(contours[i])
                    # cv2.rectangle(dst, (x,y), (x+w, y+h), (0, 255, 0), 3)
                # cv2.imshow('dst', dst)    
                # for cnt in contours:
                #     hull = cv2.convexHull(cnt)
                #     cont_img = cv2.drawContours(dst, [hull], 0, (0,0,255),2)
            cv2.putText(img, f"{curr_path} kernel size : {self.kernel_size}, {self.curr_i}/{len(self.dataset)}",(10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,0,255), 2)
            cv2.putText(img, f"median filter: {self.median_flag}",(10, 40),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,0,255), 2)
            cv2.imshow('img',img)
            


            pressed = cv2.waitKeyEx(15)
            if pressed == 27: break # Esc
            elif pressed == 56: self.change_curr_dirs(100) # 8
            elif pressed == 54: self.change_curr_dirs(1) # 6
            elif pressed == 52: self.change_curr_dirs(-1) # 4
            elif pressed == 50: self.change_curr_dirs(-100) # 2  
            elif pressed == ord('f'): self.applyfft()
            elif pressed == ord('m'): self.applymask()
            elif pressed == ord('j'): self.applymedian()
            elif pressed == ord('p'): self.change_kernel_size(5)
            elif pressed == ord('o'): self.change_kernel_size(-5)
            
parser = argparse.ArgumentParser()
parser.add_argument('--path', '-p', dest='path', required=True)
parser.add_argument('--index', '-i', dest='index',type = int, default = 0)
args = parser.parse_args()

viewer = Viewer(args.path,args.index)   
viewer.view()
