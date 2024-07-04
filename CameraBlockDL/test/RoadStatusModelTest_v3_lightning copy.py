import csv
import torch
import os
import argparse
import getpass
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms.functional import to_pil_image
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

# 기존 모듈 임포트
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from data.RoadStatusModelDS_v2 import RoadStatusDataset
from data.RoadStatusModelAT_v2 import annotate

# CSV 및 TXT 쓰기 함수 정의
def csvwriter(csv_dir, target_list):
    with open(csv_dir, 'w', newline="") as file:
        writer = csv.writer(file)
        writer.writerows(target_list)
    print(f'List Saved at {csv_dir} Succesfully')

def txtwriter(txt_dir, target_list):
    with open(txt_dir, 'w', newline="") as file:
        file.write('\n'.join(target_list))
    print(f'List Saved at {txt_dir} Succesfully')

# 사용자 이름 가져오기
username = getpass.getuser()

# 파서 정의
parser = argparse.ArgumentParser()
parser.add_argument('-update', '--update', dest='update', action='store_true')
args = parser.parse_args()

# 업데이트 옵션 처리
if args.update == True:
    nia_img_dir = f'/media/{username}/T7/2024-Summer-Internship/NIA2021'
    cbtree_img_dir = f'/media/{username}/T7/2024-Summer-Internship/벚꽃'
    clean_csv_save_dir = f'/media/{username}/T7/2024-Summer-Internship/scene/clean'
    dirty_csv_save_dir = f'/media/{username}/T7/2024-Summer-Internship/scene/dirty'
    annotate(nia_img_dir, cbtree_img_dir, clean_csv_save_dir, dirty_csv_save_dir)

# 배치 사이즈 및 학습 파라미터 설정
train_batch_sz, val_batch_sz, test_batch_sz = 32, 2, 20
epochs, learning_rate = 10, 1e-5

# 데이터셋 로드
front_path = f'/media/{username}/T7/2024-Summer-Internship'
clean_test_ds = RoadStatusDataset(front_path + '/scene/clean_test.csv')
dirty_test_ds = RoadStatusDataset(front_path + '/scene/dirty_test.csv')

test_ds = dirty_test_ds + clean_test_ds
test_dl = DataLoader(dataset=test_ds, batch_size=test_batch_sz, shuffle=True, drop_last=False)

classes = ['clean', 'dirty']

# LightningModule 정의
class RoadStatusModel(pl.LightningModule):
    def __init__(self):
        super(RoadStatusModel, self).__init__()
        self.model = torch.load('1719907458.2549872_model.pt') # 기존 모델 로드
        
    def forward(self, x):
        return self.model(x)

    def test_step(self, batch, batch_idx):
        imgs, labels, paths = batch
        output = self(imgs)
        _, preds = torch.max(output, 1)
        tp, tn, fp, fn = 0, 0, 0, 0
        false_batch = []

        for i in range(len(labels)):
            pred = preds[i].item()
            label = labels[i].item()
            path = paths[i]

            if label == 0:
                if pred == 0:
                    tp += 1
                else:
                    fn += 1
                    false_batch.append(path)
            else:
                if pred == 1:
                    tn += 1
                else:
                    fp += 1
                    false_batch.append(path)
        
        return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'false_batch': false_batch}
    
    def test_epoch_end(self, outputs):
        tp = sum([x['tp'] for x in outputs])
        tn = sum([x['tn'] for x in outputs])
        fp = sum([x['fp'] for x in outputs])
        fn = sum([x['fn'] for x in outputs])
        false_batch = [x for output in outputs for x in output['false_batch']]

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        recall = tp / (tp + fn)
        specificity = tn / (fp + tn)
        precision = tp / (tp + fp)
        f1 = 2 * precision * recall / (precision + recall)
        
        print(f'Test Finished\nTrue Positive: {tp}, True Negative: {tn}, False Positive: {fp}, False Negative: {fn}')
        print(f'Accuracy: {accuracy*100}%, Recall: {recall*100}%, Specificity: {specificity*100}%, Precision: {precision*100}%, F1: {f1}')
        
        csvwriter('result.csv', false_batch)
        txtwriter('result.txt', false_batch)

# 모델 로드
model = RoadStatusModel.load_from_checkpoint('path_to_your_checkpoint.ckpt')

# 테스트 실행
trainer = Trainer(gpus=1)
trainer.test(model, test_dataloaders=test_dl)
