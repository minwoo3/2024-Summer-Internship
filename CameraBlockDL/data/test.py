import csv
import os

with open('/home/rideflux/2024-Summer-Internship/CameraBlockDL/data/dirty_train.csv','r',newline='') as f:
    data = list(csv.reader(f))
for i in range(len(data)):
    data[i] = ''.join(data[i])
# print(data)
image_list = []
dataset_dir = '/media/rideflux/T7/2024-Summer-Internship/NIA2021/10002'
for i in range(len(data)):
    scene_abs_path = f'{dataset_dir}/image0'
    scene_image_list = os.listdir(scene_abs_path)
    for image in scene_image_list:
        if '@' not in image:
            image_list.append(image)

print(image_list.shape)
list1 = [1 for _ in range(len(image_list))]
print(list1)
# for rows in data:
#     print(''.join(rows))
# # print(data)