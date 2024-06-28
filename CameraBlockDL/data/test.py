import csv

with open('/home/rideflux/2024-Summer-Internship/CameraBlockDL/data/dirty_train.csv','r',newline='') as f:
    data = list(csv.reader(f))
for rows in data:
    print(''.join(rows))
# print(data)