import cv2
import os


import argparse
import re

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

class Viewer():
    def __init__(self, root_path):
        self.root_path = root_path # 05-08까지
        self.dirs = sorted_alphanumeric(os.listdir(root_path))[10:19]
        self.curr_i = 0
        self.set_curr_dirs()

    def set_curr_dirs(self):
        self.curr_dir = os.path.join(self.root_path, self.dirs[self.curr_i])
        self.img_paths = [os.path.join(self.curr_dir, 'camera_0', x) for x in sorted_alphanumeric(os.listdir(os.path.join(self.curr_dir, 'camera_0')))]
    
    def change_curr_dirs(self, dif):
        self.curr_i += dif
        self.curr_i %= len(self.dirs)
        self.set_curr_dirs()

    def view(self):
        idx = 0
        while True:
            idx %= len(self.img_paths)
            img = cv2.imread(self.img_paths[idx])
            cv2.putText(img, f"{os.path.basename(self.curr_dir)} {idx}",(10, 20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 0, 0), 2)
            cv2.imshow('viewer', img)

            pressed = cv2.waitKeyEx(15)
            if pressed == 27: break
            elif pressed == 96: self.change_curr_dirs(-1); idx=0
            elif pressed==9: self.change_curr_dirs(1); idx=0

            idx += 1

parser = argparse.ArgumentParser()
parser.add_argument('--path', '-p', dest='path', required=True)
args = parser.parse_args()

viewer = Viewer(args.path)   
viewer.view()
