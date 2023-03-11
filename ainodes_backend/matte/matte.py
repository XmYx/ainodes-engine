import copy
import os.path
import subprocess

import numpy as np
import torch

from ainodes_backend.matte import MattingNetwork
from ainodes_backend.poormans_wget import poorman_wget

"""try:
    # Try to run wget with the --version option
    result = subprocess.run(['wget', '--version'], capture_output=True, check=True)
    print('wget is installed')
except:
    # If wget is not found, subprocess.CalledProcessError is raised
    print('wget is not installed')
    subprocess.run(["pip", "install", "wget"])"""

class MatteInference:
    def __init__(self):
        super().__init__()
        self.model = MattingNetwork("resnet50").cuda().eval()  # or "resnet50"

        if not os.path.isfile("models/other/rvm_resnet50.pth"):
            try:
                poorman_wget("https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50.pth", "models/other/rvm_resnet50.pth")
                #subprocess.run(["wget", "-P", "models/other",
                #                "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50.pth"])
            except Exception as e:
                print("Could not load ResNet50 Matting model, please download it from:")
                print("https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50.pth")
                print("And place it in models/other")
                print(e)


        self.model.load_state_dict(torch.load("models/other/rvm_resnet50.pth"))
        self.model = torch.jit.script(self.model)
        self.model = torch.jit.freeze(self.model)

    def infer(self, frame, return_bg=False):
        # bgr = torch.tensor([.40, 1, .6]).view(3, 1, 1).cuda()  # Green background.
        rec = [None] * 4  # Initial recurrent states.
        downsample_ratio = 1.0  # Adjust based on your video.
        np_input = copy.deepcopy(frame)
        img = (np_input.astype(np.float32) / 255).transpose(2, 0, 1)[None, ...]
        img = torch.from_numpy(img).to("cuda")
        with torch.no_grad():
            # for src in DataLoader(frame):
            # print(img.shape)
            fgr, pha, *rec = self.model(img.to("cuda"), *rec, downsample_ratio)

            #Full image is torch.Tensor: img
            fgr = fgr * pha.gt(0)
            com = torch.cat([fgr, pha], dim=-3)
            com = com.mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()
            pha = 1 - pha
            bgr = img / pha.gt(0)

            com2 = torch.cat([bgr, pha], dim=-3)
            com2 = com2.mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()

            fgr = fgr.mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()
            pha = pha.mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()


        return pha[0], fgr[0], com[0], com2[0]
        # writer.write(com)  # Write frame.


