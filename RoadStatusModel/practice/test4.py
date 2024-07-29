import numpy as np
import cv2

bin_path = '1.bin'
img_path = '1.jpg'
img = cv2.imread(img_path)
# img_path = '/media/ampere_2_1/sata-ssd/camera_inference/GeneralCase/Raw/30000/camera_0/10.bin'
# img_path = '/media/ampere_2_1/sata-ssd/camera_inference/NIA2021/10002/10002_000.bin'
# img = cv2.imread(img_path)
bin = np.fromfile(bin_path, dtype = np.float16).reshape(-1,3)
# print(bin)
# bin = np.frombuffer(bin, dtype=np.int8).reshape(-1,3)
# bin = bin.reshape((930,1440))
# bin = np.array(bin,dtype=np.uint8)
ux = bin[:,0]
uy = bin[:,1]
print(ux)
ux = ux / 1440 * 1280
uy = uy / 930 * 720
print(ux)
for x, y in zip(ux,uy):
    cv2.circle(img,(int(x),int(y)),5,(0,255,0))

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# print(bin.shape)
# print(bin)
# print(len(bin))