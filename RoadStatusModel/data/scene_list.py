import csv, sys, os, getpass
username = getpass.getuser()
nas_folder_dir = f'/media/{username}/sata-ssd'

def natural_sort_key(s):
    import re
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

with open('generalcase_add.csv', 'r', newline="") as f:
    data = list(csv.reader(f))

total = []
for scene in data:
    scene_path = nas_folder_dir + scene[0]
    scene_img = os.listdir(scene_path)
    scene_img.sort(key=natural_sort_key)
    for img in scene_img:
        total.append([scene[0] + "/" + img, 1])

csv_dir = 'output.csv'
with open(csv_dir, 'w', newline="") as file:
    writer = csv.writer(file)
    writer.writerows(total)
