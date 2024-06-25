# CameraBlockDL 의 main() 역할을 하는 파일
import os
import sys
from configs.config import save_cfg, load_cfg, create_config
from CameraBlockDL.train.train import train
import CameraBlockDL.test.classification_test as cls_test
from miscs.pformat import pprint
from CameraBlockDL.model.export import export
from pygit2 import Repository

# realpath(): 현재 파일의 표준 경로 + 이름 반환
# dirname(): 입력 경로의 폴더 경로 반환
dir_path = os.path.dirname(os.path.realpath(__file__))
load_cfg_id = [
]

if __name__ == "__main__":
    # 필요 디렉토리가 있는지, 없으면 생성하는 코드
    cfgs = []
    dir_checks = [f"{dir_path}/configs/archived",
                  f"{dir_path}/test/archived",
                  f"{dir_path}/test/archived/confusion_matrix",
                  f"{dir_path}/test/archived/custom_cm",
                  f"{dir_path}/test/archived/latency_results",
                  f"{dir_path}/model/archived",
                  f"{dir_path}/train/archived",
                  f"{dir_path}/model/exported",]

    #isdir(): 폴더 유무 판단
    for dir_check in dir_checks:
        if not os.path.isdir(dir_check):
            pprint(f"no archive file in '{dir_check}', creating directory", ["HEADER"])
            os.mkdir(dir_check)

    for cfg_id in load_cfg_id:
        cfgs.append(load_cfg(cfg_id))

    m = input("create config? [Y/n]: \n")

    while m in ["y", "Y", "yes", "Yes"] or m == "":
        backbone = input("provide backbone name [beomjun_model_middle, mobilenet_v2, efficientnet_b0, googlenet, no_pool, etc.]: \n")
        #??
        repo = Repository('.').head.shorthand
        tag = input("provide tag name (ex: PER-XYZ), default 0: \n")
        if backbone == "":
            backbone = "beomjun_model_middle"
        if tag == "":
            tag = '0'
        tag = f'{repo}_{tag}'

        cfg = create_config(backbone=backbone, tag=tag)
        cfgs.append(cfg)

        m = input("create config? [Y/n]: \n")



    for cfg in cfgs:
        if cfg.train.do:
            train(cfg)

        if cfg.test.do:
            cls_test.python_test(cfg)
            # latency_test.test(cfg)

        if cfg.export.do:
            export(cfg)
        #
        # cls_test.python_test(cfg)
        # data_cfg = load_cfg("ulsan_test")
        # cls_test.new_data_test(model_cfg=cfg, data_cfg=data_cfg)
