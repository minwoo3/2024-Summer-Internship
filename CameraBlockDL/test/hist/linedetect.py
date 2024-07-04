import cv2
import numpy as np
import matplotlib.pylab as plt

#--① 큐브 영상 읽어서 HSV로 변환
img = cv2.imread("/home/rideflux/다운로드/2024-04-03-15-19-18_solati_v5_5_3-24/camera_0/30.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#--② 색상별 영역 지정
white1 = np.array([0, 0, 200])
white2 = np.array([180, 25, 255])
red1 = np.array([0, 50,50])
red2 = np.array([15, 255,255])
red3 = np.array([165, 50,50])
red4 = np.array([180, 255,255])
yellow1 = np.array([20, 50,50])
yellow2 = np.array([35, 255,255])
gray1 = np.array([0, 0, 50])
gray2 = np.array([180, 50, 200])
brown1 = np.array([10, 50, 50])
brown2 = np.array([20, 255, 200])
black1 = np.array([0, 0, 0])
black2 = np.array([180, 255, 50])
green1 = np.array([35, 50, 50])
green2 = np.array([85, 255, 255])


# --③ 색상에 따른 마스크 생성
mask_white = cv2.inRange(hsv, white1, white2)
mask_red1 = cv2.inRange(hsv, red1, red2)
mask_red2 = cv2.inRange(hsv, red3, red4)
mask_yellow = cv2.inRange(hsv, yellow1, yellow2)
mask_gray = cv2.inRange(hsv, gray1, gray2)
mask_brown = cv2.inRange(hsv, brown1, brown2)
mask_black = cv2.inRange(hsv, black1, black2)
mask_green = cv2.inRange(hsv, green1, green2)

#--④ 색상별 마스크로 색상만 추출
res_white = cv2.bitwise_and(img, img, mask=mask_white)
res_red1 = cv2.bitwise_and(img, img, mask=mask_red1)
res_red2 = cv2.bitwise_and(img, img, mask=mask_red2)
res_red = cv2.bitwise_or(res_red1, res_red2)
res_yellow = cv2.bitwise_and(img, img, mask=mask_yellow)
res_gray = cv2.bitwise_and(img, img, mask=mask_gray)
not_gray = cv2.bitwise_not(img, img, mask_gray)
res_brown = cv2.bitwise_and(img, img, mask=mask_brown)
res_black = cv2.bitwise_and(img, img, mask=mask_black)
res_green = cv2.bitwise_and(img, img, mask=mask_green)

res_combined1 = cv2.bitwise_or(res_red, res_yellow)
res_combined2 = cv2.bitwise_or(res_combined1, res_white)

_, bin_gray = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY_INV)

#--⑤ 결과 출력
# imgs = {'original': img, 'white':res_white,'red':res_red, 'yellow':res_yellow, 'gray': res_gray, 'brown': res_brown, 'black': res_black, 'green': res_green,  'combined':res_combined2}
# for i, (k, v) in enumerate(imgs.items()):
#     plt.subplot(2,5, i+1)
#     plt.title(k)
#     plt.imshow(v[:,:,::-1])
#     plt.xticks([]); plt.yticks([])
# plt.show()
plt.subplot(1,2,1)
plt.imshow(res_gray)

plt.subplot(1,2,2)
plt.imshow(bin_gray)
plt.show()