# 2024-Summer-Internship
연속된 카메라 이미지를 기반으로 노면 오염도를 판단하는 모델 및 알고리즘 개발

#### Conda 이용해서 설치
```shell
conda create -n ${name} python=3.7
conda activate ${name}
conda install pytorch torchvision torchaudio -c pytorch -c nvidia
pip install -r requirements.txt
```

#### train
```shell
argument: -m model, -c ckpt name, -t transform type
ex) python3 train.py -m 'cnn' -c 'CNNModule_014_2' -t 'mask'
```

#### test(inference)
```shell
argument: -m model, -c ckpt name, -t transform type
ex) python3 test.py -m 'cnn' -c 'CNNModule_014_2' -t 'mask'
*test set 변경 시: data/datamodule.py line 28 에서 'test.csv' 변경
*crop 적용/미적용 시: data/dataset.py 에서 crop 여부 설정
```

#### CAM(class activation map)
```shell
argument: -m model, -c ckpt name, -t transform type -p path -i index
ex) python3 cam.py -m 'cnn' -c 'CNNModule_014_2' -t 'mask' -p '/media/ampere_2_1/T7/2024-Summer-Internship/scene/CNNModule_014_2/test9.csv' -i 3000
*index는 필요시 지정
*wasd 혹은 2/4/6/8 키로 이미지 이동
```

#### Main Checkpoints
```shell
no mask, BCE Loss(no weight) : CNNModule_008_2.ckpt
4channel, BCE Loss(no weight) : CNNModule_013_4.ckpt
4channel, BCE Loss(weight 3.16) : CNNModule_013_2.ckpt
4channel, BCE Loss(weight 3.16), crop : CNNModule_014.ckpt
4channel, BCE Loss(weight 1.77), crop : CNNModule_014_2.ckpt    <---- Latest
*weight 수정: model/module.py에서 train_step 함수 수정
*mask 수정: train.py 컴파일 시 -t argument 입력 X
```

