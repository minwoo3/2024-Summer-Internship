import os
import torch
from CameraBlockDL.miscs.pformat import pprint
from CameraBlockDL.model.pl_module import get_module
from CameraBlockDL.configs.config import load_cfg
import time
from torchsummary import summary


# print(torch.device)

total_memory = torch.cuda.get_device_properties(0).total_memory
# print("total memort: ", total_memory)
# less than 0.5 will be ok:
# tmp_tensor = torch.empty(int(total_memory * 0.0299), dtype=torch.int8, device='cuda')
# del tmp_tensor

# this allocation will raise a OOM:
# torch.empty(total_memory // 2, dtype=torch.int8, device='cuda')

# print(torch.cuda.list_gpu_processes())
# print(torch.cuda.memory_allocated() / (1024 ** 3), torch.cuda.memory_reserved() / (1024 ** 3))

curr_dir_path = os.path.dirname(os.path.realpath(__file__))
# ckpt_path = "/home/beomjun/Perception-TrainingSW/Train/TrafficlightDetection/saved_model/Trafficlight_model-epoch=09-val_loss=0.00_focal_loss_no_alpha_sequential_split.ckpt"




def export(cfg, verbose=True, cuda=True):
    save_dir = cfg.export.save_dir
    model_name = cfg.export.model_name
    if model_name == "":
        model_name = cfg.id + ".pt"
    onnx_name = "camera_block_" + cfg.id + ".onnx"


    CROPPED_IMG_HEIGHT = cfg.image.CROPPED_IMG_HEIGHT
    CROPPED_IMG_WIDTH = cfg.image.CROPPED_IMG_WIDTH


    print(torch.device)



    model = get_module(cfg, "export")
    sample_input = torch.rand(1, 3, CROPPED_IMG_HEIGHT,CROPPED_IMG_WIDTH)

    if cuda:
        model = model.cuda()
        sample_input = sample_input.cuda()
        model_name = "cuda_" + model_name
    else:
        model_name = "cpu_" + model_name


    model.eval()
    summary(model, (3, CROPPED_IMG_HEIGHT, CROPPED_IMG_WIDTH), batch_size=10)
    block_script_module = model.to_torchscript(method="trace", example_inputs=sample_input)
    # tl_script_module = torch.jit.trace(model, sample_input)

    torch.jit.save(block_script_module, os.path.join(save_dir, model_name))




    torch.onnx.export(model,  # model being run
                      sample_input,  # model input (or a tuple for multiple inputs)
                      f"{save_dir}/{onnx_name}",  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}})


    with torch.no_grad():
        print(torch.cuda.list_gpu_processes())
        print(torch.cuda.memory_allocated() / (1024 ** 3), torch.cuda.memory_reserved() / (1024 ** 3))
        torch.cuda.empty_cache()


    if verbose:
        pprint(f"[export] successfully exported {model_name}", options=["OKGREEN"])

if __name__=='__main__':
    cfg_id = "2023-08-18-18-16-49_beomjun_model_middle_PER-370-3"
    cfg = load_cfg(cfg_id)
    export(cfg, cuda=True)
