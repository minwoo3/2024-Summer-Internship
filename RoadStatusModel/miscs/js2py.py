# json에서 읽어온 edict 파일의 값을 파이썬 타입으로 변환

from easydict import EasyDict as edict
from copy import deepcopy
# from TrafficlightClassificationDL.miscs.pformat import pprint
from RoadStatusModel.miscs.pformat import pprint


def convert_string(str_val, sformat):
    str_val = \
        str_val.replace(" ", "")
    if sformat == "f":
        return float(str_val)

    elif sformat == "b":
        if str_val.lower() == "true":
            return True
        else:
            return False

    elif sformat == "d":
        return int(str_val)

    elif sformat == "s":
        return str_val

    else:
        pprint(f"[convert_string] Couldm't understand given format {sformat}!", ["bold", "fail"])
        raise ()


def convert(ed):
    ned = deepcopy(ed)
    for k, val in ed.items():
        if type(val) == edict:
            ned[k] = convert(val)

        elif type(val) == str:
            if "%" in val:
                str_vals, formats = val.split("%")
                n = len(formats)
                str_vals_split = str_vals.split(",")
                if len(str_vals) == 0:
                    ned[k] = []
                    continue
                elif n == 1:
                    py_vals = convert_string(str_vals, formats)
                    ned[k] = py_vals
                    continue

                elif n == len(str_vals_split):
                    py_vals = []
                    for i in range(n):
                        py_vals.append(convert_string(str_vals_split[i], formats[i]))
                    ned[k] = py_vals

                else:
                    pprint(f"format size ({len(formats)}) does not match string ({len(str_vals_split)})",
                           ["bold", "fail"])
                    raise ()

    return ned