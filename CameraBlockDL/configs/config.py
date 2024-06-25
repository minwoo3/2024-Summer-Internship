import os
import sys
import json
from easydict import EasyDict as edict
from copy import deepcopy
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))) # import st_intern_hyunkunkim sys path
from CameraBlockDL.miscs.pformat import pprint
import CameraBlockDL.miscs.js2py as js2py
# 쉘 명령 등 다른 프로세스를 실행하고 출력 결과를 가져올 수 있게 해주는 라이브러리
import subprocess
from CameraBlockDL.configs.templates import default_config

"""
templates.py 를 활용하여 쉽게 config edict 객체를 생성.
저장 및 불러오기는 json 형식을 활용하되 불러올 때 miscs의 js2py 를 활용.
js2py 를 활용하면 json 의 string을 python 객체로 자연스럽게 변환할 수 있음.
"""

file_path = os.path.realpath(__file__)
CameraBlockDL_dir = "/".join(file_path.split("/")[:-2])
file_dir = f"{CameraBlockDL_dir}/configs"
data_dir = f"{CameraBlockDL_dir}/data/datas"
annotations_dir = f"{CameraBlockDL_dir}/data/datas/annotations"

def save_cfg(ed, verbose=True):
    with open(f"{file_dir}/archived/{ed.id}.json", "w") as f:
        json.dump(ed, f, indent=4)

    if verbose:
        pprint(f"[save_cfg] config {ed.id} saved", ["okgreen"])

    return None

def load_cfg(cid, verbose=True, pythonic=True):
    if f"{cid}.json" not in os.listdir(f"{file_dir}/archived"):
        if verbose:
            pprint(f"[load_cfg] config {cid} not found!", ["bold", "fail"])
        return None

    with open(f"{file_dir}/archived/{cid}.json", "r") as f:
        ed = edict(json.load(f))

    if verbose:
        pprint(f"[load_cfg] config {cid} loaded", ["okgreen"])

    if pythonic:
        ed = js2py.convert(ed)

    return ed

# create ID per configs, datetime (임시로 date로 해놓음),
# 추가적인 정보는 additional_info 에 옵션으로 줄 수 있음.
def make_id(additional_info=None):
    id = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    if additional_info is not None:
        id += f"_{additional_info}"

    return id

def create_config(manual=True, verbose=True, backbone="beomjun_model_middle", tag=None):
    config_dict = deepcopy(default_config)
    config_dict.backbone = backbone
    config_dict.train.filename = "-".join([backbone, config_dict.train.filename])
    additional_info = f"{backbone}_{tag}"
    config_id = make_id(additional_info=additional_info)
    config_dict.id = config_id

    save_cfg(config_dict, verbose=False)

    if manual:
        # subprocess.call(args, *, stdin=None, stdout=None, stderr=None, shell=False, timeout=None)
        # args : 쉘에 입력할 명령어 문자열을 공백 문자로 나눈 문자열 리스트
        # shell : 별도의 서브 쉘을 실행하고 그 위에서 명령을 실행할 지 여부를 지정, shell=True 로 사용하는 경우 args는 리스트가 아닌 문자열 형태로 쓰는게 좋음.
        subprocess.call(f"gedit {file_dir}/archived/{config_id}.json", shell=True)
        config_dict = load_cfg(config_id, verbose=False)

    save_cfg(config_dict, verbose=verbose)

    return config_dict

