import os.path
import subprocess

import numpy as np
import torch

from ainodes_backend.matte import MattingNetwork


try:
    # Try to run wget with the --version option
    result = subprocess.run(['wget', '--version'], capture_output=True, check=True)
    print('wget is installed')
except subprocess.CalledProcessError:
    # If wget is not found, subprocess.CalledProcessError is raised
    print('wget is not installed')
    subprocess.run(["pip", "install", "wget"])

class MatteInference:
    def __init__(self):
        super().__init__()
        self.model = MattingNetwork("resnet50").cuda().eval()  # or "resnet50"

        if not os.path.isfile("models/other/rvm_resnet50.pth"):
            try:
                subprocess.run(["wget", "-P", "models/other",
                                "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50.pth"])
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
        img = (frame.astype(np.float32) / 255).transpose(2, 0, 1)[None, ...]
        img = torch.from_numpy(img).to("cuda")
        with torch.no_grad():
            # for src in DataLoader(frame):
            # print(img.shape)
            fgr, pha, *rec = self.model(img.to("cuda"), *rec, downsample_ratio)
            fgr = fgr * pha.gt(0)
            # com = torch.cat([fgr, pha], dim=-3)
            fgr = fgr.mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()
            pha = pha.mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()
        return pha[0], fgr[0]
        # writer.write(com)  # Write frame.



"""import subprocess

# Install wget using pip
subprocess.run(["pip", "install", "wget"])

# Download the file using wget
subprocess.run(["wget", "-P", "models/other", "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50.pth"])
"""