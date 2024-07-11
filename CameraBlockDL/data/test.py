import csv
import os

# def csvwriter(csv_dir, target_list):
#     with open(csv_dir, 'w', newline="") as file:
#         writer = csv.writer(file)
#         writer.writerows(target_list)

# with open('30000.csv','r',newline='') as f:
#     data = list(csv.reader(f))
# for i in range(len(data)):
#     data[i] = ''.join(data[i])
    # total = []
    # for row in data:
    #     total.append([row[0],row[8]])

# csvwriter('30000_copy.csv',total)
# with open('30000_copy.csv','r',newline='') as f:
#     data = list(csv.reader(f))
# print(data[0][1])

path = os.listdir('/home/rideflux/Public/TrainingData/GeneralCase/Raw/30000/camera_0')
print(path)


# image_list = []
# dataset_dir = '/media/rideflux/T7/2024-Summer-Internship/NIA2021/10002'
# for i in range(len(data)):
#     scene_abs_path = f'{dataset_dir}/image0'
#     scene_image_list = os.listdir(scene_abs_path)
#     for image in scene_image_list:
#         if '@' not in image:
#             image_list.append(image)

# print(image_list.shape)
# list1 = [1 for _ in range(len(image_list))]
# print(list1)
# # for rows in data:
# #     print(''.join(rows))
# # # print(data)