from io import BytesIO

import numpy as np
import torch
from .models import *
from .datasets import *
from torchvision import transforms as T
from torchvision import io


import torch
import argparse
import yaml
import math
from torch import Tensor
from torch.nn import functional as F
from pathlib import Path
from torchvision import io
from torchvision import transforms as T
from .models import *
from .datasets import *
from .utils.utils import timer
from .utils.visualize import draw_text
from PIL import Image


class SemSegModel():


    def __init__(self) -> None:
        # inference device cuda or cpu
        self.device = torch.device("cuda")

        # get dataset classes' colors and labels
        self.palette = eval('ADE20K').PALETTE
        self.labels = eval('ADE20K').CLASSES

        # initialize the model and load weights and send to device
        self.model = eval('SegFormer')(
            backbone='MiT-B3',
            num_classes=150
        )
        self.model.load_state_dict(torch.load('models/other/segformer.b3.ade.pth', map_location='cpu'))
        self.model = self.model.to(self.device)
        self.model.eval()

        # preprocess parameters and transformation pipeline
        self.size = [512, 512]
        self.tf_pipeline = T.Compose([
            T.Lambda(lambda x: x / 255),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            T.Lambda(lambda x: x.unsqueeze(0))
        ])

    def preprocess(self, image: Tensor) -> Tensor:
        #image = image.permute(2, 0, 1)
        #print(image.shape)

        #H, W = image.shape[0,1]
        # console.print(f"Original Image Size > [red]{H}x{W}[/red]")
        # scale the short side of image to target size
        #scale_factor = self.size[0] / min(H, W)
        #nH, nW = round(H * scale_factor), round(W * scale_factor)
        # make it divisible by model stride
        #nH, nW = int(math.ceil(nH / 32)) * 32, int(math.ceil(nW / 32)) * 32
        # console.print(f"Inference Image Size > [red]{nH}x{nW}[/red]")
        # resize the image
        image = T.Resize((512, 512))(image)
        # divide by 255, norm and add batch dim
        image = self.tf_pipeline(image).to(self.device)
        return image

    def postprocess(self, orig_img: Tensor, seg_map: Tensor, overlay: bool) -> Tensor:
        # resize to original image size
        seg_map = F.interpolate(seg_map, size=orig_img.shape[-2:], mode='bilinear', align_corners=True)
        # get segmentation map (value being 0 to num_classes)
        seg_map = seg_map.softmax(dim=1).argmax(dim=1).cpu().to(int)

        # convert segmentation map to color map
        seg_image = self.palette[seg_map].squeeze()
        if overlay:
            seg_image = (orig_img.permute(1, 2, 0) * 0.4) + (seg_image * 0.6)
        print(seg_image.shape)
        image, masks = draw_text(seg_image, seg_map, self.labels)
        return image, masks

    @torch.inference_mode()
    @timer
    def model_forward(self, img: Tensor) -> Tensor:
        return self.model(img)

    def predict(self, image, overlay=False):
        buffer = BytesIO()
        image = Image.fromarray(image)
        image.save("test.png", format='PNG')
        png_data = "test.png"

        image = io.read_image(png_data)
        #image = Image.fromarray(image)
        img = self.preprocess(image)
        seg_map = self.model_forward(img)
        seg_map, masks = self.postprocess(image, seg_map, True)
        return seg_map, masks
    def infer(self, image):

        # resize
        # Transpose the dimensions of the image to CHW format
        image = image.permute(2, 0, 1)

        # Apply the center crop transform
        image = T.CenterCrop((512, 512))(image)

        # Transpose the dimensions of the image back to HWC format
        #image = image.permute(1, 2, 0)
        # scale to [0.0, 1.0]
        image = image.float() / 255
        # normalize
        image = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image)
        # add batch size
        image = image.unsqueeze(0)
        image.shape
        with torch.inference_mode():
            seg = self.model(image)
        seg.shape
        seg = seg.softmax(1).argmax(1).to(int)
        seg.unique()
        palette = eval('ADE20K').PALETTE
        seg_map = palette[seg].squeeze().to(torch.uint8)
        if seg_map.shape[2] != 3: seg_map = image.permute(1, 2, 0)
        seg_map = seg_map.numpy()
        return seg_map
