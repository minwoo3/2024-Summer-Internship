import os,csv

# scene_list = []
# with open('generalcase_add.csv','r') as f:
#     data = list(csv.reader(f))    
#     for scene in data:
#         scene_list.append(scene[0])
total = []

front_dir = '/media/ampere_2_1/T7/2024-Summer-Internship/NIA2021'
scene_list = list(os.listdir(front_dir))
first_img = []
for scene in scene_list:
    first_img.append([f'{front_dir}/{scene}/image0/{scene}_000.jpg',0])

front_dir = '/media/ampere_2_1/T7/2024-Summer-Internship/벚꽃'
day_list = list(os.listdir(front_dir))
for day in day_list:
    scene_dir = front_dir + '/' + day
    scene_list = list(os.listdir(scene_dir))
    for scene in scene_list:
        first_img.append([f'{front_dir}/{day}/{scene}/camera_0/0.jpg',0])

front_dir = '/media/ampere_2_1/sata-ssd/GeneralCase/Raw'
scene_list = list(os.listdir(front_dir))
for scene in scene_list:
    first_img.append([f'{front_dir}/{scene}/image0/0.jpg',0])

# for scene in scene_list:
#     img_list = list(os.listdir(front_dir))
#     for img in img_list:
#         total.append([scene+'/'+img,0])

with open('result.csv','w',newline ='') as f:
    writer = csv.writer(f)
    writer.writerows(first_img)
print(len(first_img))

# for scene in data:
#     front_dir = '/home/ampere_2_1'
