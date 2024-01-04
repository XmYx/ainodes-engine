import os
import cv2
import torch
import argparse
from torch.nn import functional as F
import warnings

from backend_helpers.torch_helpers.RIFE.RIFE import Model


warnings.filterwarnings("ignore")


class RIFEModel():
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.torch_options_applied = None
        self.apply_torch_options()
        self.load_RIFE()
        

    def apply_torch_options(self):
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
        self.torch_options_applied = True
        
    def load_RIFE(self):
        self.model = Model()
        self.model.load_model("train_log", -1)
        self.model.eval()
        self.model.device()
        print("Loaded v3.x HD model.")
    
    def open_exr(self, image1, image2):
        if image1.endswith('.exr') and image2.endswith('.exr'):
            img0 = cv2.imread(image1, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
            img1 = cv2.imread(image2, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
            img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(self.device)).unsqueeze(0)
            img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(self.device)).unsqueeze(0)
            return img0, img1
    
    def open_jpg_png(self, image1, image2):

        img0 = cv2.imread(image1, cv2.IMREAD_UNCHANGED)
        img1 = cv2.imread(image2, cv2.IMREAD_UNCHANGED)
        img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(self.device) / 255.).unsqueeze(0)
        img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(self.device) / 255.).unsqueeze(0)
        return img0, img1
    
    def open_np_array(self, image1, image2):
        img0 = (torch.tensor(image1.transpose(2, 0, 1)).to(self.device) / 255.).unsqueeze(0)
        img1 = (torch.tensor(image2.transpose(2, 0, 1)).to(self.device) / 255.).unsqueeze(0)
        return img0, img1

    def infer(self, image1, image2, exp, ratio, rthreshold, rmaxcycles):
        
        img0, img1 = self.open_np_array(image1, image2)

        n, c, h, w = img0.shape
        ph = ((h - 1) // 32 + 1) * 32
        pw = ((w - 1) // 32 + 1) * 32
        padding = (0, pw - w, 0, ph - h)
        img0 = F.pad(img0, padding)
        img1 = F.pad(img1, padding)
        with torch.no_grad():
            if ratio:
                img_list = [img0]
                img0_ratio = 0.0
                img1_ratio = 1.0
                if ratio <= img0_ratio + rthreshold / 2:
                    middle = img0
                elif ratio >= img1_ratio - rthreshold / 2:
                    middle = img1
                else:
                    tmp_img0 = img0
                    tmp_img1 = img1
                    for inference_cycle in range(rmaxcycles):
                        middle = self.model.inference(tmp_img0, tmp_img1)
                        middle_ratio = ( img0_ratio + img1_ratio ) / 2
                        if ratio - (rthreshold / 2) <= middle_ratio <= ratio + (rthreshold / 2):
                            break
                        if ratio > middle_ratio:
                            tmp_img0 = middle
                            img0_ratio = middle_ratio
                        else:
                            tmp_img1 = middle
                            img1_ratio = middle_ratio
                img_list.append(middle)
                img_list.append(img1)
            else:
                img_list = [img0, img1]
                for i in range(exp):
                    tmp = []
                    for j in range(len(img_list) - 1):
                        mid = self.model.inference(img_list[j], img_list[j + 1])
                        tmp.append(img_list[j])
                        tmp.append(mid)
                        print("appending RIFE frame")
                    tmp.append(img1)
                    img_list = tmp
            #print(len(img_list))
        
        if not os.path.exists('output'):
            os.mkdir('output')
        np_images = []
        for i in range(len(img_list)):

            np_images.append((img_list[i][0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])
            #if img[0].endswith('.exr') and img[1].endswith('.exr'):
            #    cv2.imwrite('output/img{}.exr'.format(i), (img_list[i][0]).cpu().numpy().transpose(1, 2, 0)[:h, :w], [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
            #else:
            #cv2.imwrite('output/img{}.png'.format(i), (img_list[i][0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])
        #print(len(np_images))
        return np_images