import torch


class VAEDecodeModule(torch.nn.Module):
    def __init__(self, module, decode):
        super().__init__()
        self.module = module
        self.decode = decode

    def forward(self, samples):
        return self.decode(samples)
