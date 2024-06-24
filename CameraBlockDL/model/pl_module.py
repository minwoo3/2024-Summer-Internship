import os
import sys
import torch
import pytorch_lightning as pl
from CameraBlockDL.model.nn_model import _model
from CameraBlockDL.miscs.pformat import pprint

from torch import argmax
from torch.optim import Adam, AdamW
from torch.nn import CrossEntropyLoss
import CameraBlockDL.test.presentation as present
import pickle
from time import time

"""
BlockageClassificationModule은 딥러닝 네트워크 (nn.Module) 뿐 아니라,
학습시, 검증시, 테스트시, 테스트 완료 시 어떻게 할지에 대한 정보도 담겨 있음.
어떠한 손실함수를 쓸 지 부터, 학습 도중 어떠한 값을 로그에 남길지,
테스트 완료시에는 어떠한 시각화를 할 지 등을 여기에 지정할 수 있음.
이후 Trainer 객체가 학습, 검증, 테스트를 할 경우 여기에 있는 절차데로 하게 됨.
"""


class BlockageClassificationModule(pl.LightningModule):
    def __init__(self, cfg={}):
        super().__init__()
        self.cfg = cfg
        num_of_classes = cfg.labels.num_of_classes
        loss_weights = cfg.labels.loss_weights
        self.latency = []
        # self.net = Net(num_of_classes, cfg.backbone)
        if hasattr(cfg, 'input_shape'):
            self.net = _model(backbone=cfg.backbone, num_of_classes=num_of_classes, batch_norm=True, pretrained=True,
                              progress=True, input_shape=cfg.input_shape)
        else:
            self.net = _model(backbone=cfg.backbone, num_of_classes=num_of_classes, batch_norm=True, pretrained=True,
                              progress=True)
        self.test_y, self.test_y_hat = [], []
        self.loss_weights = torch.Tensor(loss_weights)
        self.val_epoch_outputs = []
        self.test_epoch_outputs = []

    def forward(self, x):
        x = self.net(x)
        return x

    def configure_optimizers(self):
        # return Adam(self.parameters(), lr=1e-3)
        optimizer = AdamW(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)
        return [optimizer], [scheduler]

    def xy_from_batch(self, dataset_batch):
        x_list, y_list = [], []
        if isinstance(dataset_batch[0], list):
            for batch in dataset_batch:
                xd, yd = batch
                x_list.append(xd)
                y_list.append(yd)
        else:
            x_list.append(dataset_batch[0])
            y_list.append(dataset_batch[1])

        x, y = torch.cat(x_list, 0), torch.cat(y_list, 0)

        # print(x.shape)
        # print(y.shape)

        return x, y

    def training_step(self, dataset_batch, batch_idx):
        x, y = self.xy_from_batch(dataset_batch)
        logits = self(x)

        if self.loss_weights != None:
            weights = self.loss_weights.to(self.device)

        loss = CrossEntropyLoss(weight=weights, label_smoothing=0.1)(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, dataset_batch, batch_idx, dataloader_idx=0):
        x, y = self.xy_from_batch(dataset_batch)

        logits = self(x)

        # loss = CrossEntropyLoss()(logits, y)
        if self.loss_weights != None:
            weights = self.loss_weights.to(self.device)
        loss = CrossEntropyLoss(weight=weights, label_smoothing=0.1)(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.val_epoch_outputs.append(loss)
        return loss

    def on_validation_epoch_start(self):
        self.val_epoch_outputs = []

    def on_validation_epoch_end(self):
        try:
            #todo. check sanity
            total_loss = torch.cat((torch.stack(self.val_epoch_outputs[0]), torch.stack(self.val_epoch_outputs[1])), dim=0)
        except:
            total_loss = torch.stack(self.val_epoch_outputs)
        self.log('val_loss_total', torch.mean(total_loss))

    def test_step(self, dataset_batch, batch_idx, dataloader_idx=0):
        x, y = self.xy_from_batch(dataset_batch)
        start_time = time()
        logits = self(x)
        # loss = CrossEntropyLoss()(logits, y)
        if self.loss_weights != None:
            weights = self.loss_weights.to(self.device)
        loss = CrossEntropyLoss(weight=weights, label_smoothing=0.1)(logits, y)

        preds = argmax(logits, dim=1)
        end_time = time()
        self.latency.append(end_time - start_time)
        acc = (preds == y).float().mean()
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        self.test_epoch_outputs.append([preds, y])
        return [preds, y]

    def on_test_epoch_start(self):
        self.test_epoch_outputs = []

    def on_test_epoch_end(self):
        print("test_epoch_end", len(self.test_epoch_outputs))
        total_preds, total_ys = [], []
        # todo. 테스트 데이터셋의 수 자동으로 카운트 하도록 변
        num_test_dataset = 0
        for cfg_dataset in self.cfg.datasets:
            if 'test' in self.cfg.datasets[cfg_dataset].annotation_files.keys():
                num_test_dataset += 1

        if num_test_dataset > 1:
            print("More than one test data")
            for dataset in range(num_test_dataset):
                for output in self.test_epoch_outputs[dataset]:
                    # 제 생각에는 위에 수정한 것처럼 하면 이 부분이 필요없을 것 같은데 확인 부탁드려요.
                    # while len(output) == 1:
                    #     output = output[0]
                    try:
                        preds, ys = output
                    except:
                        continue
                    total_preds.extend(preds.cpu().numpy())
                    total_ys.extend(ys.cpu().numpy())
        else:
            print("Only one test data")
            for output in self.test_epoch_outputs:
                # 제 생각에는 위에 수정한 것처럼 하면 이 부분이 필요없을 것 같은데 확인 부탁드려요.
                # while len(output) == 1:
                #     output = output[0]
                # print(output.shape)
                try:
                    preds, ys = output
                except:
                    continue
                total_preds.extend(preds.cpu().numpy())
                total_ys.extend(ys.cpu().numpy())
        present.make_confusion_matrix(total_ys, total_preds, self.cfg)


def get_module(cfg: object, mtype: object = "test") -> object:
    num_of_classes = cfg.labels.num_of_classes
    loss_weights = cfg.labels.loss_weights
    checkpoint_path = cfg[mtype].checkpoint_path
    if checkpoint_path == "best":
        cfg_ckp_dir = f"{cfg.train.dirpath}/{cfg.id}"
        checkpoint_path = f"{cfg_ckp_dir}/{os.listdir(cfg_ckp_dir)[-1]}"
    module = BlockageClassificationModule.load_from_checkpoint(cfg=cfg,
                                                              num_of_classes=num_of_classes,
                                                              loss_weights=torch.Tensor(loss_weights),
                                                              checkpoint_path=checkpoint_path)
    return module