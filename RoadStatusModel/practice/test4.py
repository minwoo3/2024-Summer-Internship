import numpy as np
import cv2

# img_path = '/media/ampere_2_1/T7/2024-Summer-Internship/벚꽃/04-03/2024-04-03-10-45-13_solati_v5_6_1837-1858/camera_0/0.jpg'
# img = cv2.imread(img_path)
bin = np.fromfile('/home/rideflux/Public/LaneLineCamera/lane_label_image0/10002/10002_000.bin', dtype = bool)
print(bin)
print(len(bin))