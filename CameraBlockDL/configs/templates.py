import os
from easydict import EasyDict as edict

"""
configs.py 에서 필요한 cfg edict 객체를 빠르게 만들기 위해 템플릿을 모아놓은 파일
많이 사용되고 반복해서 쓸 것 같은 옵션은 클래스로 지정
디폴트 옵션을 담은 default_config 객체 생성
"""

file_path = os.path.realpath(__file__)
tl_classification_dir = "/".join(file_path.split("/")[:-2])
file_dir = f"{tl_classification_dir}/configs"
data_dir = f"/media/hyunkun/ReT7/CameraBlockedDataset"
annotations_dir = f"{data_dir}/labels"
model_dir = f"{tl_classification_dir}/model"

#### Common configurations ####
class Commons:
    image = edict({
        "CROPPED_IMG_HEIGHT": 88,
        "CROPPED_IMG_WIDTH": 120,
    })

    labels = edict({
        "num_of_classes": 2, # normal, blocked
        "loss_weights": [1.0, 1.0]
    })


#### Transformation edicts ####
class Transforms:
    Normalize = edict({
            "do": True,  # one of True, False, Train, Valid
            "args": "",
            "kwargs": {"mean": [0.485, 0.456, 0.406],
                        "std": [0.229, 0.224, 0.225]}
    })

    RandomCrop = edict({
        "do": True,  # one of True, False, Train, Valid
        "args": [Commons.image.CROPPED_IMG_HEIGHT, Commons.image.CROPPED_IMG_WIDTH],  # have to be same as "CROPPED_IMG_HEIGHT, CROPPED_IMG_WIDTH"
        "kwargs": {}
    })

    Resize = edict({
        "do": True,
        "args": [Commons.image.CROPPED_IMG_HEIGHT, Commons.image.CROPPED_IMG_WIDTH],
        "kwargs": {"antialias": True}
    })
    ColorJitter = edict({
        "do": True,
        "args": "",
        "kwargs": {"brightness": 0.2,
                   "contrast": 0.2,
                   "saturation": 0.2,
                   "hue": 0.05}
    })
    RandomAffine = edict({
        "do": True,
        "args": "",
        "kwargs": {"degrees": 1,
                   "translate": (0.1, 0.1),
                   "scale": (0.9, 1.1)}
    })



camera_block_config = edict({
    "name": "camera_block",
    "from_npimg": False,
    "annotation_files": {
        "train": f"{annotations_dir}/camera_block_train.csv", #todo. change annotation file
        "val": f"{annotations_dir}/camera_block_val.csv",
        "test": f"{annotations_dir}/camera_block_test.csv",
    },
    "data_dir": f"{data_dir}/imgs/pt_format",  # where data.pt files are located
    # "data_dir": f"{pt_dir}",  # 데이터를 ssd에 저장하고 다닐 때, pt_dir 활용


    "transforms": edict({"Resize": Transforms.Resize,
                         "Normalize": Transforms.Normalize,
                         "RandomAffine": Transforms.RandomAffine,
                         "ColorJitter": Transforms.ColorJitter}),

    "transforms_order": ["Resize", "ColorJitter", "Normalize", "RandomAffine"]
})




default_config = edict({
    "id": "1990-01-01-00-00-00", # 자동 생성
    "backbone": "beomjun_model_middle",
    "dataset_names": ["camera_block"],
    "datasets": {"camera_block": camera_block_config},

    "image": Commons.image,
    "labels": Commons.labels,

    "data_loader": {
        "batch_size": 256,
        "num_workers": 6,
        "pin_memory": True,
        "shuffle": True
    },

    "train": {
        "do": True,
        "min_epochs": 10,
        "max_epochs": 100,
        "patience": 3,
        "monitor": "val_loss_total",
        "dirpath": f"{model_dir}/archived",  # 경로가 하드 코딩 되어있어서 수정
        "filename": "tl_classification-{epoch:02d}-{val_loss:.4f}", # save naming format
        "shuffle": True
    },

    "test": {
        "do": True,
        "checkpoint_path": "best",  # Either best, None or path to the checkpoint you wish to test.
    },

    "export":{
        "do": True,
        "checkpoint_path": "best",
        "save_dir": f"{model_dir}/exported",
        "model_name": "",
    },
    "notes": {},
    "seed": 73,  # torch.manual_seed(73),
})



