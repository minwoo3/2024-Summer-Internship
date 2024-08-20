import csv, os

def csvwriter(csv_dir, target_list):
    with open(csv_dir, 'w', newline="") as file:
        writer = csv.writer(file)
        writer.writerows(target_list)
    print(f'List Saved at {csv_dir} Successfully')

annotation_file = 'test6.csv'
with open(annotation_file,'r',newline='') as f:
    data = list(csv.reader(f))

result = []
cnt = 0
for img in data:
    if cnt % 10 == 0:
        result.append([img[0],img[1]])
    cnt += 1
csvwriter('nia.csv',result)