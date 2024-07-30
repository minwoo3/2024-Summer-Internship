import numpy as np
import cv2

curb_path = '11856_000.bin'
lane_path = '11856_000_2.bin'
img_path = '11856_000.jpg'
img = cv2.imread(img_path)
width, height = 1280, 720
img = cv2.resize(img,(width,height))

test_img = np.fromfile("/media/rideflux/MJ_SSD/Boundarylabel/11856/11856_000.bin", dtype = bool).reshape(930, 1440)



# curb = np.fromfile(curb_path, dtype = np.float16).reshape(-1,3)
curb = np.fromfile(curb_path, dtype = int).reshape(2, -1).T
# lane = np.fromfile(curb_path, dtype = int).reshape(-1,2)
# lane = np.fromfile(lane_path, dtype = np.float16).reshape(-1,3)

curb_x, curb_y = curb[:,0], curb[:,1]

curb_x, curb_y = curb_x / 1440 * width, curb_y / 930 * height

# lane_x, lane_y = lane[:,0], lane[:,1]
# lane_x, lane_y = lane_x / 1440 * width, lane_y / 930 * height

# curb_x, curb_y = [int(x) for x in curb_x], [int(y) for y in curb_y]
curb_x, curb_y = np.clip(curb_x.astype(int), 0, width - 1), np.clip(curb_y.astype(int), 0, height - 1)
# lane_x, lane_y = [int(x) for x in lane_x], [int(y) for y in lane_y]

for x, y in zip(curb_x, curb_y):
    cv2.circle(img,(int(x),int(y)),1,(0,255,0),thickness= -1)

# for x, y in zip(lane_x, lane_y):
#     cv2.circle(img,(int(x),int(y)),1,(255,0,0),thickness= -1)

#마스크는 단일 채널
mask = np.zeros((height,width),dtype=np.uint8)
mask[curb_y, curb_x] = 255
# mask[lane_y, lane_x] = 255
cv2.imshow('img',img)
# mask2 = cv2.resize(mask,(40,22))

kernel_size = 5
kernel = np.ones((kernel_size, kernel_size), np.uint8)
mask2 = cv2.dilate(mask, kernel, iterations=1)
cv2.imshow('img2',mask2)
# img = cv2.resize(img,(40,22))

# extracted = cv2.bitwise_and(img,img,mask=mask)

# cv2.imshow('img',mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
