import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import lightning as pl
from easydict import EasyDict as edict
import numpy as np
from CameraBlockDL.configs.config import save_cfg
from tqdm import tqdm
"""
BlockDataset 이 가장 Low level 정보를 다룸, 파일 경로에서 읽기, 라벨 값 읽기 등.
BlockDataset 은 DataLoader 객체 안으로 들어가서 사용됨. 
DataLoader 는 배치 사이즈, 핀메모리 등 데이터 외적으로 불러오는데 필요한 정보를 담음
DataModule 객체는 여러 DataLoader 객체를 담음. train, val, test 상황에 따라 적절한 데이터 로더들의 리스트를 리턴함.
"""

class BlockDataset(Dataset):
    def __init__(self, annotations_file, data_path, transform=None, kfold=1, kidx=1, random_state=100, from_npimg=False, num_of_classes=2):
        """ Traffic Light Dataset class \n
        데이터에 대한 가장 디테일한 정보를 다룸. \n
        파일 경로, 라벨 값 csv 파일 등의 정보를 저장하고 사용함.

        :param annotations_file: csv file path for the dataset
        :type annotations_file: str
        :param data_path: directory path where dataset is located
        :param from_npimg: 일반적인 이미지를 numpy에서 불로오는 방식인지
        :type from_npimg: bool
        """
        self.df = pd.read_csv(annotations_file)

        self.from_npimg = from_npimg

        self.df = self.df.sample(frac=1, random_state=random_state).reset_index(drop=True)

        self.data_path = data_path
        self.transform = transform
        if hasattr(self.df, 'label'):
            labels = self.df.iloc[:, 1]
        else:
            try:
                labels = self.df.iloc[:, 0].apply(lambda f: int(f.split("/")[-2]))
            except:
                raise()
        num_of_labels = labels.value_counts()
        self.loss_weights = np.array([num_of_labels[i] if i in num_of_labels else 0 for i in range(num_of_classes)])


    def __len__(self):
        return len(self.df)

    def get_origin(self, idx):
        data_name = self.df.iloc[idx, 0]

        if self.from_npimg:
            if hasattr(self.df, 'label'):
                images, labels = load_npimg(os.path.join(self.data_path, data_name), get_label=False, get_origin=True)
                labels = self.df.iloc[idx, 1]
            else:
                images, labels = load_npimg(os.path.join(self.data_path, data_name), get_label=True, get_origin=True)
        else:
            images, labels = torch.load(os.path.join(self.data_path, data_name))
        return images, labels

    def __getitem__(self, idx):
        data_name = self.df.iloc[idx]['img_file']

        if self.from_npimg:
            if hasattr(self.df, 'label'):
                images, labels = load_npimg(os.path.join(self.data_path, data_name), get_label=False)
                labels = self.df.iloc[idx, 1]
                labels = torch.tensor(labels, dtype=torch.uint8)
            else:
                try:
                    images, labels = load_npimg(os.path.join(self.data_path, data_name), get_label=True)
                except:
                    raise()
        else:
            try:
                images, labels = torch.load(os.path.join(self.data_path, data_name))
            except:
                images = torch.zeros([3, 180, 320])
                labels = torch.tensor(1)
                print(data_name)
                # raise()


        # TODO. Normalization 과정 두 번 겪는건 아닌지 체크하기.
        images = images / 255
        images = images.float()
        if data_name == 'cut_mix_nia':
            print('nia')
        if self.transform != None:
            images = self.transform(images)


        return images, labels.long()


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        """ DataModule 객체 생성 (중간에 영어 주석 포기함)
        여러 데이터셋의 스플릿 (학습, 검증, 테스트) 에 따라 각각 데이터 로더를 생성,
        학습 부터 테스트 까지 필요한 모든 데이터 로더를 다루는 객체.
        또한 모든 데이터셋으로 부터 빈도수를 카운트하여 loss_weight를 1/sqrt(빈도수) 로 설정해 주는 역할을 함.

        :param cfg: Configuration EasyDict object
        :type cfg: edict
        """

        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.data_loader.batch_size

        self.loaders = edict({})
        self.loss_weights = np.array(cfg.labels.loss_weights)

        for split in ["train", "val", "test"]:
            loaders = []
            for dsname in self.cfg.datasets.keys():
                if split not in self.cfg.datasets[dsname].annotation_files:
                    continue
                loader, weights = get_loader(self.cfg, dsname=dsname, dstype=split, give_weights=True)
                loaders.append(loader)
                self.loss_weights += weights

            self.loaders[split] = loaders

        # Loss weights 는 data_loader에서 집계한 빈도수 제곱근에 반비례,
        ## 빈도수 높으면 적은 가중치를 받도록 됨
        self.loss_weights = 1 / np.sqrt(self.loss_weights)

        # 현재 학습에 사용되는 loss_weights를 다시 저장
        # json에 저장 하려면 리스트로 변환 필요
        cfg.labels.loss_weights = list(self.loss_weights)
        save_cfg(cfg, verbose=False)

    def train_dataloader(self):
        return self.loaders.train

    def val_dataloader(self):
        return self.loaders.val

    def test_dataloader(self):
        # TODO. 모든 데이터셋을 train, val, test 3분류로 변환
        return self.loaders.test


def load_npimg(data_name, get_label=True, get_origin=False):
    if not os.path.isfile(data_name):
        print(data_name)
        assert os.path.isfile(data_name)
    cvimg = cv2.imread(f'{data_name}')
    if get_label:
        label = int(data_name.split('/')[-2])
    else:
        label = None

    if get_origin:
        return cvimg, label
    npimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
    npimg = np.swapaxes(npimg, 0, 1)
    npimg = np.swapaxes(npimg, 0, 2)

    img = torch.from_numpy(npimg)
    if label is not None:
        label = torch.tensor(label, dtype=torch.uint8)

    return img, label

def get_transform(cfg, dsname="NIA2020"):
    """ Returns the composed transformations described in cfg

    :param cfg: The configuration file from configs module
    :param dsname: The dataset name
    :type cfg: edict
    :type dsname: str
    :returns: composed transformations for the dataset
    :rtype: transforms.Compose
    """
    transform_list = []

    # Iterate over transforms dictionary (key is transform_name, values are transform_options)
    if hasattr(cfg.datasets[dsname], 'transforms_order'):
        transform_name_options = [(tr, cfg.datasets[dsname].transforms[tr]) for tr in cfg.datasets[dsname].transforms_order]
    else:
        transform_name_options = cfg.datasets[dsname].transforms.items()

    for transform_name, transform_options in transform_name_options:
        if transform_options.do:
            transform_args = transform_options.args
            transform_kwargs = transform_options.kwargs

            if transform_args != '':
                # TODO. 현재 args를 1개만 받도록 됨, *args로 할 경우 버그 생김
                transform_list.append(getattr(transforms, transform_name)(transform_args, **transform_kwargs))
            else:
                transform_list.append(getattr(transforms, transform_name)(**transform_kwargs))
        else:
            continue

    composed_transform = transforms.Compose(transform_list)
    return composed_transform


def get_blockdataset(cfg, dsname="rf2020", dstype="train"):
    from_npimg = False
    if hasattr(cfg, 'from_npimg'):
        if cfg.from_npimg:
            from_npimg = True
    if hasattr(cfg.datasets[dsname], 'from_npimg'):
        if cfg.datasets[dsname].from_npimg:
            from_npimg = True

    transform = get_transform(cfg, dsname=dsname)

    block_dataset = BlockDataset(cfg.datasets[dsname].annotation_files[dstype], cfg.datasets[dsname].data_dir, transform, random_state=cfg.seed, from_npimg=from_npimg, num_of_classes=cfg.labels.num_of_classes)
    return block_dataset


def get_loader(cfg, dsname="rf2020", dstype="train", give_weights=False):
    # dataloader 부를 때, **kwargs 로 하면 에러 나서 그냥 원래대로 해놓음
    block_dataset = get_blockdataset(cfg, dsname=dsname, dstype=dstype)
    batch_size = cfg.data_loader.batch_size
    num_workers = cfg.data_loader.num_workers
    pin_memory = cfg.data_loader.pin_memory
    if dstype == "train":
        shuffle = cfg.data_loader.shuffle
    else:
        shuffle = False
    dataloader = DataLoader(dataset=block_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle = shuffle, persistent_workers=True)

    if give_weights:
        return dataloader, block_dataset.loss_weights
    return dataloader

