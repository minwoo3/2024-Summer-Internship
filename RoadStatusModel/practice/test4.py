import numpy as np
import cv2
import sys, os
import PIL.Image as pil
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from data.dataset_v4 import RoadStatusDataset

annotation_file = '/media/ampere_2_1/T7/2024-Summer-Internship/scene/CNNModule_009/train.csv'
dataset = RoadStatusDataset(annotation_file= annotation_file, transform_flag = 'mask')
print(len(dataset))