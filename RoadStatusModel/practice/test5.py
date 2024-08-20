import csv, os

def csvwriter(csv_dir, target_list):
    with open(csv_dir, 'w', newline="") as file:
        writer = csv.writer(file,quoting=csv.QUOTE_MINIMAL)
        writer.writerows(target_list)
    print(f'List Saved at {csv_dir} Successfully')

annotation_file = 'test6.csv'
with open(annotation_file,'r',newline='') as f:
    data = list(csv.reader(f))


dir = '/media/ampere_2_1/T7/2024-Summer-Internship/NIA2021/'
nia = []
for scene in data:
    scene_dir = dir+scene[0]+'/image0'
    img_list = os.listdir(scene_dir)
    cnt = 0
    for img in img_list:
        if cnt % 10 ==4:
            x = f'/NIA2021/{scene[0]}/image0/{img}'
            nia.append([x])
        cnt += 1

print(len(nia))
    
csvwriter('nia.csv',nia)