import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_histogram_peaks(image, num_bins=256):
    hist = cv2.calcHist([image], [0], None, [num_bins], [0, 256])
    peaks = np.argsort(hist.ravel())[::-1]
    return peaks[:3]  # 상위 3개 색상 피크를 반환

def create_mask_from_peak(image, peak, range_size=40):
    lower_bound = np.array([max(0, peak - range_size)] * 3)
    upper_bound = np.array([min(255, peak + range_size)] * 3)
    mask = cv2.inRange(image, lower_bound, upper_bound)
    return mask

# 이미지 파일 경로를 절대 경로로 변경
img_path = "/home/rideflux/다운로드/2024-04-03-15-19-18_solati_v5_5_3-24/camera_0/30.jpg"  # 여기에 올바른 경로를 입력하세요
img = cv2.imread(img_path)

# 이미지가 정상적으로 읽혔는지 확인
if img is None:
    print("Error: Could not read the image.")
    exit()

# 이미지를 그레이스케일로 변환
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 히스토그램의 상위 3개 피크 값을 가져옴
peaks = get_histogram_peaks(gray)

# 각 피크에 대해 마스크 생성
masks = [create_mask_from_peak(gray, peak) for peak in peaks]

# 각 마스크를 이미지에 적용
result_images = [cv2.bitwise_and(img, img, mask=mask) for mask in masks]

# 결과 출력
imgs = {'original': img}
for i, (mask, result) in enumerate(zip(masks, result_images)):
    imgs[f'mask_{i}'] = mask
    imgs[f'result_{i}'] = result

plt.figure(figsize=(15, 10))
for i, (k, v) in enumerate(imgs.items()):
    plt.subplot(2, len(imgs)//2 + 1, i + 1)
    plt.title(k)
    if len(v.shape) == 2:
        plt.imshow(v, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(v, cv2.COLOR_BGR2RGB))
    plt.xticks([]); plt.yticks([])
plt.show()
