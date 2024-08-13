import csv, cv2, torch, sys, os
import argparse, getpass
import numpy as np
from PIL import Image
import torch.nn.functional as F
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.module import CNNModule, ResnetModule
from data.dataset import RoadStatusDataset
def csvreader(csv_dir):
    with open(csv_dir, 'r', newline='') as f:
        data = list(csv.reader(f))
    paths, labels = [], []
    for path, label in data:
        paths.append(path)
        labels.append(int(label))
    print('Read CSV Successfully')
    return paths, labels

class Viewer():
    def __init__(self, csv_path, index, model, transform_flag):
        self.csv_path, self.curr_i, self.transform_flag = csv_path, index, transform_flag
        self.t7_dir = f'/media/{username}/T7/2024-Summer-Internship'
        self.sata_dir = f'/media/{username}/sata-ssd'

        with open(csv_path,'r',newline='') as f:
            data = list(csv.reader(f))

        self.img_path, self.img_label= [], []
        for path, label in data:
            self.img_path.append(path)
            self.img_label.append(int(label))

        self.classes = ['clean','dirty']
        opt, batch_size = 1e-5, 16

        self.dataset = RoadStatusDataset(annotation_file= args.path, transform_flag= args.transform)
        example_img, _, _ = self.dataset[0]
        self.img_height, self.img_width = example_img.shape[-2:]  # (height, width)

        if model in ['cnn','CNN']:
            ssd_dir = f'{self.t7_dir}/checkpoint/cnn'
            self.module = CNNModule.load_from_checkpoint(f'{ssd_dir}/{args.checkpoint}.ckpt',
                                                    img_width=self.img_width, img_height=self.img_height, 
                                                    opt=opt, ckpt_name = args.checkpoint, loss_weight_mode = args.weight)
        elif model in ['resnet','res','ResNet']:
            ssd_dir = f'{self.t7_dir}/checkpoint/resnet'
            self.module = ResnetModule.load_from_checkpoint(f'{ssd_dir}/{args.checkpoint}.ckpt',
                                                    opt = opt, strict = False)
        else:
            raise ValueError("Invalid model name. Choose from ['cnn', 'CNN', 'resnet', 'res', 'ResNet']")
    
        self.module_name = self.module.__class__.__name__
        self.mask, self.hsv, self.cam = False, False, True

    def change_curr_dirs(self, dif):
        self.curr_i += dif
        if self.curr_i == len(self.img_path):
            print('end of list')
            self.curr_i -= dif
        elif self.curr_i == 0:
            print('first of list')
            self.curr_i += dif

    def applymask(self):
        if self.mask == False: self.mask = True
        else: self.mask = False
    
    def applycam(self):
        if self.cam == False: self.cam = True
        else: self.cam = False

    # def applyhsvmask(self):
    #     if self.hsv == False: self.hsv = True
    #     else: self.hsv = False

    def drawCAM(self, width, height):
        activation_map = self.module.featuremap.squeeze().cpu()
        class_weights_gap = F.adaptive_avg_pool2d(activation_map,(1,1)).squeeze()
        cam = torch.zeros(activation_map.shape[1:], dtype=torch.float32)
        for i in range(len(class_weights_gap)):
            cam += class_weights_gap[i]*activation_map[i,:,:]
        cam = F.relu(cam)
        # print(max(map(max,cam)).item())
        # print(min(map(min,cam)).item())
    
        cam = cam - cam.min()
        cam = cam / cam.max()
        cam = (cam > 0.6) * cam

        cam = cam.detach().numpy()
        cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((width, height), Image.Resampling.LANCZOS)) / 255.0
        return cam_resized

    def hsvmask(self,img):
        # print(img.shape)
        b = img[:,:,0]
        g = img[:,:,1]
        r = img[:,:,2]
        b_eq = cv2.equalizeHist(b)
        g_eq = cv2.equalizeHist(g)
        r_eq = cv2.equalizeHist(r)
        adjust = cv2.merge((b_eq,g_eq,r_eq))
        hsv = cv2.cvtColor(adjust, cv2.COLOR_BGR2HSV)

        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        result = cv2.bitwise_and(cv2.cvtColor(img,cv2.COLOR_BGR2RGB),cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                                , mask = cv2.bitwise_not(mask))
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        return result

    def view(self):
        while True:
            curr_img, curr_label, curr_path = self.dataset[self.curr_i]
            
            if 'NIA' in curr_path or '벚꽃' in curr_path:
                show_img = cv2.imread(self.t7_dir + curr_path)
            elif 'GeneralCase' in curr_path:
                show_img = cv2.imread(self.sata_dir + curr_path)
            show_img = cv2.resize(show_img, (self.img_width, self.img_height))
            # print(show_img.shape)
            # curr_img[:,:self.img_height//2,:] = 0
            
            
            
            output = self.module.model(curr_img.unsqueeze(0))
            
            pred = torch.sigmoid(output.squeeze(1))
            
            pred_class = (pred>0.3).long()
            

            if self.cam == True:
                cam = self.drawCAM(self.img_width, self.img_height)
                heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_MAGMA)
                heatmap = np.float32(heatmap) / 255
                overlay = heatmap + np.float32(show_img / 255)
                show_img = overlay / np.max(overlay)

            if self.mask == True:
                mask = np.array(curr_img[3,:,:]).astype(np.uint8)
                show_img_uint8 = (show_img * 255).astype(np.uint8)
                show_img = cv2.bitwise_and(show_img_uint8, show_img_uint8, mask=mask)
                show_img = show_img.astype(np.float32) / 255

            

            cv2.putText(show_img, f"{args.checkpoint}/ {curr_path}  {self.curr_i}/{len(self.img_path)}",(10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,0), 2)
            cv2.putText(show_img, f"Label: {self.classes[curr_label]} / Pred: {self.classes[pred_class]} / {pred.item():.2%}",(10, 40),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,0), 2)
            cv2.imshow('overlay',show_img)

            pressed = cv2.waitKeyEx(15)
            if pressed == 27: break # Esc
            elif pressed == 56: self.change_curr_dirs(100) # 8
            elif pressed == 54: self.change_curr_dirs(1) # 6
            elif pressed == 52: self.change_curr_dirs(-1) # 4
            elif pressed == 50: self.change_curr_dirs(-100) # 2
            # elif pressed == ord('0'): self.change_curr_dirs(-100) # 2
            # elif pressed == ord('0'): self.change_curr_dirs(-100) # 2          
            elif pressed == ord('m'): self.applymask()
            elif pressed == ord('c'): self.applycam()
            # elif pressed == ord('h'): self.applyhsvmask()


username = getpass.getuser()
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', dest='path', required=True)
parser.add_argument('-m', '--model', dest='model', action = 'store')
parser.add_argument('-c', '--ckpt', dest='checkpoint', action = 'store')
parser.add_argument('-i', '--index', dest='index',type = int, default = 0)
parser.add_argument('-t', '--transform', dest='transform', action = 'store')
parser.add_argument('-w', '--weight', dest = 'weight', action = 'store', default = None)
args = parser.parse_args()

torch.cuda.empty_cache()

viewer = Viewer(args.path,args.index,args.model, args.transform)   
viewer.view()
