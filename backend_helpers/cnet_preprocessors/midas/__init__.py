import math
import os

import cv2
import numpy as np
import torch
from PIL import Image

from einops import rearrange
from .api import MiDaSInference
import torchvision.transforms as T

from .midas.dpt_depth import DPTDepthModel
from .midas.transforms import Resize, PrepareForNet, NormalizeImage
import requests
from tqdm import tqdm


class MidasDetector:
    def __init__(self):
        self.model = MiDaSInference(model_type="dpt_hybrid").cuda()
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.midas_transform = T.Compose([
            Resize(
                384, 384,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet()
        ])
        self.device = "cuda"


    def __call__(self, input_image, a=np.pi * 2.0, bg_th=0.1):
        #print("Using", a, bg_th)
        assert input_image.ndim == 3
        image_depth = input_image
        with torch.no_grad():
            image_depth = torch.from_numpy(image_depth).float().cuda()
            image_depth = image_depth / 127.5 - 1.0
            image_depth = rearrange(image_depth, 'h w c -> 1 c h w')
            depth = self.model(image_depth)[0]

            depth_pt = depth.clone()
            depth_pt -= torch.min(depth_pt)
            depth_pt /= torch.max(depth_pt)
            depth_pt = depth_pt.cpu().numpy()
            depth_image = (depth_pt * 255.0).clip(0, 255).astype(np.uint8)

            depth_np = depth.cpu().numpy()
            x = cv2.Sobel(depth_np, cv2.CV_32F, 1, 0, ksize=3)
            y = cv2.Sobel(depth_np, cv2.CV_32F, 0, 1, ksize=3)
            z = np.ones_like(x) * a
            x[depth_pt < bg_th] = 0
            y[depth_pt < bg_th] = 0
            normal = np.stack([x, y, z], axis=2)
            normal /= np.sum(normal ** 2.0, axis=2, keepdims=True) ** 0.5
            normal_image = (normal * 127.5 + 127.5).clip(0, 255).astype(np.uint8)

            return depth_image, normal_image
    def predict(self, prev_img_cv2) -> torch.Tensor:
        # convert image from 0->255 uint8 to 0->1 float for feeding to MiDaS
        img_midas = prev_img_cv2.astype(np.float32) / 255.0
        img_midas_input = self.midas_transform({"image": img_midas})["image"]
        # MiDaS depth estimation implementation
        sample = torch.from_numpy(img_midas_input).float().to(self.device).unsqueeze(0)
        if self.device == torch.device("cuda"):
            sample = sample.to(memory_format=torch.channels_last)
            sample = sample.half()
        with torch.no_grad():
            midas_depth = self.deforum_midas.forward(sample)
        midas_depth = torch.nn.functional.interpolate(
            midas_depth.unsqueeze(1),
            size=img_midas.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        midas_depth = midas_depth.cpu().numpy()
        # MiDaS makes the near values greater, and the far values lesser. Let's reverse that and try to align with AdaBins a bit better.
        midas_depth = np.subtract(50.0, midas_depth)
        midas_depth = midas_depth / 19.0
        depth_map = midas_depth
        depth_map = np.expand_dims(depth_map, axis=0)
        depth_tensor = torch.from_numpy(depth_map).squeeze().to(self.device)
        return depth_tensor
    def load_midas(self, half_precision=True):
        models_path = "models/midas"
        os.makedirs(models_path, exist_ok=True)
        if not os.path.exists(os.path.join(models_path, 'dpt_large-midas-2f21e586.pt')):
            print("Downloading dpt_large-midas-2f21e586.pt...")
            poorman_wget("https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt", models_path)

        self.deforum_midas = DPTDepthModel(
            path=f"{models_path}/dpt_large-midas-2f21e586.pt",
            backbone="vitl16_384",
            non_negative=True,
        )


        #gs.models["midas_model"].eval()
        if half_precision and self.device == torch.device("cuda"):
            self.deforum_midas.to(memory_format=torch.channels_last)
            self.deforum_midas.half()
        self.deforum_midas.to(self.device)

        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        self.midas_transform = T.Compose([
            Resize(
                384, 384,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet()
        ])

def poorman_wget(url, filename):
    ckpt_request = requests.get(url, stream=True)
    request_status = ckpt_request.status_code

    total_size = int(ckpt_request.headers.get("Content-Length", 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit="B", unit_scale=True)

    # inform user of errors
    if request_status == 403:
        raise ConnectionRefusedError("You have not accepted the license for this model.")
    elif request_status == 404:
        raise ConnectionError("Could not make contact with server")
    elif request_status != 200:
        raise ConnectionError(f"Some other error has ocurred - response code: {request_status}")

    # write to model path
    with open(filename, 'wb') as model_file:
        for data in ckpt_request.iter_content(block_size):
            model_file.write(data)
            progress_bar.update(len(data))

    progress_bar.close()

def wget_headers(url):
    r = requests.get(url, stream=True, headers={'Connection':'close'})
    return r.headers

def wget_progress(url, filename, length=0, chunk_size=8192, callback=None):

    one_percent = int(length) / 100
    next_percent = 1

    with requests.get(url, stream=True) as r:

        r.raise_for_status()
        downloaded_bytes = 0
        callback(next_percent)
        with open(filename, 'wb') as f:
            try:
                for chunk in r.iter_content(chunk_size=8192):
                    # If you have chunk encoded response uncomment if
                    # and set chunk_size parameter to None.
                    #if chunk:
                    f.write(chunk)
                    downloaded_bytes += chunk_size
                    if downloaded_bytes > next_percent * one_percent:
                        next_percent += 1
                        callback(next_percent)
            except Exception as e:
                print('error while writing download file: ', e)
