import random
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
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

lane = []
with open('10007_152.json', 'r') as fcc_file:
    data = json.load(fcc_file)
    for i in range(len(data['annotations'])):
        annt = data['annotations'][i]
        if annt['class'] == 'lane':
            lane.append(annt['polyline'])
img = cv2.imread('10007_152.jpg')
# img = cv2.imread('8.jpg')
img = cv2.resize(img,(2880,1860))
i = 4
cx, cy = lane[i][0], lane[i][1]
nx, ny = lane[i][2], lane[i][3]
poly = np.polyfit([cx,nx],[cy,ny],1)

i = 5
cx2, cy2 = lane[i][0], lane[i][1]
nx2, ny2 = lane[i][2], lane[i][3]
poly2 = np.polyfit([cx2,nx2],[cy2,ny2],1)

x = [cx, nx]
y = [cy, ny]
x_pred = (np.array(y)-poly2[1])/poly2[0]
# y_pred = np.array(x)*poly[0] + poly[1]
# print(y_pred[0])
cv2.circle(img,(cx, cy),10,(255,255,0))
cv2.circle(img,(nx, ny),10,(255,255,0))
cv2.circle(img,(cx2, cy2),10,(255,255,0))
cv2.circle(img,(nx2, ny2),10,(255,255,0))
cv2.circle(img,(int(x_pred[0]), cy),10,(0,255,0))
cv2.circle(img,(int(x_pred[1]), ny),10,(0,255,0))
for i in range(10):
    rx, ry = random.randint(cx, int(x_pred[0])), random.randint(ny, cy)
    cv2.circle(img,(rx, ry), 5, thickness= -1,color = (0,255,0))

pts1 = [[nx, ny], [int(x_pred[1]), ny],[cx, cy], [int(x_pred[0]), cy]]
ptf_img, transform_mat = ptf(pts1,img)
ptf_img = cv2.resize(ptf_img,(1280,720))


ptf_img = cv2.cvtColor(ptf_img, cv2.COLOR_BGR2GRAY)
fft = np.fft.fft2(ptf_img)
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
# print(np.max(inv_fft))
# print(np.min(inv_fft))
# img = cv2.resize(img,(1280,720))


# img2 = np.fromfile("/home/rideflux/Public/LaneLineCamera/lane_label_image0/10007/10007_152.bin", dtype = bool).reshape(930, 1440)
# uy, ux = img2.nonzero()

# ux = ux / 1440 * 1280
# uy = uy / 930 * 720

# ux, uy = np.clip(ux.astype(int), 0, 1280 - 1), np.clip(uy.astype(int), 0, 720 - 1)

# img[uy, ux, 0] = 255
# img[uy, ux, 1] = 0
# img[uy, ux, 2] = 0

# cv2.imshow('a',img)

plt.subplot(221)
plt.imshow(ptf_img, cmap='gray')

plt.subplot(222)
plt.imshow(mag_spectrum, cmap='gray')

plt.subplot(223)
plt.imshow(mag_spectrum_2, cmap='gray')


plt.subplot(224)
plt.imshow(inv_fft, cmap='gray')
plt.show()


# cv2.imshow('crop',ptf_img)
# # cv2.imshow('fft', inv_fft)

# cv2.waitKey(0)
# cv2.destroyWindow('img')
