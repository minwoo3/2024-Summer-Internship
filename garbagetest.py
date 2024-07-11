import cv2
import numpy as np

img = cv2.imread('/media/rideflux/T7/2024-Summer-Internship/벚꽃/04-03/2024-04-03-15-19-18_solati_v5_5_17-38/camera_0/31.jpg')
# img = cv2.imread('/media/rideflux/T7/2024-Summer-Internship/NIA2021/10002/image0/10002_000.jpg')
img = cv2.resize(img, (1280, 720))
img = cv2.fastNlMeansDenoisingColored(img,None,10,10)
# edges = cv2.Canny(img,10,150)
# lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=200)

# road_mask = np.zeros_like(img)
# if lines is not None:
#     for line in lines:
#         x1, y1, x2, y2 = line[0]
#         if y1 > img.shape[0] // 2 and y2 > img.shape[0] // 2:  # 이미지 하단 부분에 있는 직선만 고려
#             cv2.line(road_mask, (x1, y1), (x2, y2), 255, 2)

# # road_only = cv2.bitwise_and(img, road_mask)
# # cv2.imshow('original',img)
# combined_img = cv2.addWeighted(img,1,road_mask,1,0)
# cv2.imshow('roadmask',road_mask)
# cv2.imshow('masked',combined_img)
# cv2.imshow('edge',edges)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
adjust = cv2.equalizeHist(gray)
denoised = cv2.fastNlMeansDenoising(adjust, None, 10)

size = 15  # 블러 크기
kernel = np.zeros((size, size))
kernel[int((size - 1)/2), :] = np.ones(size)
kernel = kernel / size

blurred = cv2.filter2D(denoised, -1, kernel)
shadow_removal = cv2.absdiff(denoised, blurred)

dx = cv2.Sobel(shadow_removal, cv2.CV_32F, 1, 0)
dy = cv2.Sobel(shadow_removal, cv2.CV_32F, 0, 1)

mag = cv2.magnitude(dx, dy)
mag = np.clip(mag, 0, 255).astype(np.uint8)

dst = np.zeros(img.shape[:2], np.uint8)
dst[mag>150] = 255

kernel = np.ones((1,1), np.uint8)
dilated_image = cv2.dilate(dst, kernel, iterations = 1)
lines = cv2.HoughLinesP(dilated_image, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=200)

road_mask = np.zeros_like(img)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if y1 > img.shape[0] // 2 and y2 > img.shape[0] // 2:
            cv2.line(road_mask, (x1, y1), (x2, y2), 255, 2)

cv2.imshow('original',img)
cv2.imshow('img2',shadow_removal)
cv2.imshow('img',road_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()