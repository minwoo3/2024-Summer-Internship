import numpy as np
import cv2
import sys, os
import PIL.Image as pil
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from data.dataset_v4 import RoadStatusDataset

lane_bin = '0.bin'
curb_bin = '0_curb.bin'
img = '0.jpg'

def openbin(src):
    ux, uy = src[:,0],src[:,1]
    ux, uy = ux / 1440 * 1280, uy / 930 * 720
    ux, uy = np.clip(ux.astype(int), 0, 1280 - 1), np.clip(uy.astype(int), 0, 720 - 1)
    return ux, uy

# lane: 차선 inference, curb: 연석 inference
lane = np.fromfile(lane_bin,dtype = np.float16).reshape(-1,3)
lane_x, lane_y = openbin(lane)
curb = np.fromfile(curb_bin, dtype = np.float16).reshape(-1,3)
curb_x, curb_y = openbin(lane)

# float16 타입 mask, 0 or 1
mask = np.zeros((720, 1280), dtype = np.uint8)
mask[lane_y, lane_x] = 1
mask[curb_y, curb_x] = 1
kernel_size = 50
kernel = np.ones((kernel_size, kernel_size), np.uint8)
mask = cv2.dilate(mask, kernel, iterations=1)

img = pil.open(img)
img = np.array(img)
img = img.astype(np.float16)

# 4채널로 합치기
combined = np.zeros((4, 720, 1280), dtype=np.float16)
combined[:3, :, :] = img.transpose(2, 0, 1)
combined[3, :, :] = mask

print('img.shape: ',img.shape)
print('mask.shape: ',mask.shape)
print('combined.shape: ',combined.shape)

combined_img = combined[:3, :, :].transpose(1, 2, 0).astype(np.uint8)
combined_mask = (combined[3, :, :] * 255).astype(np.uint8)

# 이미지 및 마스크 출력
cv2.imshow('img', combined_img)
for x, y in zip(lane_x, lane_y):
    cv2.circle(combined_img, (x,y),3,(0,255,0),-1)
cv2.imshow('mask', combined_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()