import csv
import numpy as np
import os
def csvwriter(csv_dir, target_list):
    # if not os.path.isdir(csv_dir):
        
    with open(csv_dir, 'w', newline="") as file:
        writer = csv.writer(file)
        writer.writerows(target_list)

total_csv = 'sorted_data.csv'
dirty_scene_csv = 'nia_dirty_scene.csv'
clean_scene_csv = 'nia_clean_scene.csv'

with open(total_csv,'r',newline='') as f:
    total = list(csv.reader(f))

clean, dirty = [], []
with open(dirty_scene_csv,'r',newline='') as f:
    data = list(csv.reader(f))
    dirty = [row[0] for row in data]
with open(clean_scene_csv,'r',newline='') as f:
    data = list(csv.reader(f))
    clean = [row[0] for row in data]
dirty_train,dirty_val,dirty_test = 0.8, 0.1, 0.1
clean_train,clean_val,clean_test = 0.8, 0.1, 0.1
# print(len(clean))

clean_random_choice = list(np.random.choice(clean, 50, replace= False))
# print(len(dirty))
# print(len(clean_random_choice))
# print(len(dirty), len(clean), len(clean_random_choice))
# list(np.random.choice(total, int(len(total)*train_ratio), replace = False))
dirty_train_set = list(np.random.choice(dirty, int(len(dirty)*dirty_train), replace = False))
dirty_rest_set = [x for x in dirty if x not in dirty_train_set]
dirty_val_set = list(np.random.choice(dirty_rest_set, int(len(dirty)*dirty_val), replace = False))
dirty_test_set = [x for x in dirty_rest_set if x not in dirty_val_set]
print(len(dirty_train_set),len(dirty_val_set),len(dirty_test_set))
clean_train_set = list(np.random.choice(clean_random_choice, int(len(clean_random_choice)*clean_train), replace = False))
clean_rest_set = [x for x in clean_random_choice if x not in clean_train_set]
clean_val_set = list(np.random.choice(clean_rest_set, int(len(clean_random_choice)*clean_val), replace = False))
clean_test_set = [x for x in clean_rest_set if x not in clean_val_set]
print(len(clean_train_set),len(clean_val_set),len(clean_test_set))
# print(type(clean_train_set[0]))
train, val, test = [], [], []

for i in range(len(total)):
    scene, _, img = total[i][0].split('/')
    # print(scene)
    if scene in clean_train_set or scene in dirty_train_set:
        train.append(total[i])
    elif scene in clean_val_set or scene in dirty_val_set:
        val.append(total[i])
    elif scene in clean_test_set or scene in dirty_test_set:
        test.append(total[i])

print(len(train), len(val), len(test))

csvwriter('train.csv',train)
csvwriter('val.csv',val)
csvwriter('test.csv',test)
