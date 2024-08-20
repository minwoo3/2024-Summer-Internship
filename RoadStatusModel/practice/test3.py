import csv

annotation_file = 'nia.csv'
with open(annotation_file,'r',newline='') as f:
    data = list(csv.reader(f))

print(len(data))
print(data[0][0].split("/"))