# code originally taken from: https://github.com/ChenyangSi/FreeU (under MIT License)

import torch
import copy


def Fourier_filter(x, threshold: int, scale: float):
    # FFT
    x_freq = torch.fft.fftn(x.float(), dim=(-2, -1))
    x_freq = torch.fft.fftshift(x_freq, dim=(-2, -1))

    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W), device=x.device)

    crow, ccol = H // 2, W // 2
    mask[
        ..., crow - threshold : crow + threshold, ccol - threshold : ccol + threshold
    ] = scale
    x_freq = x_freq * mask

    # IFFT
    x_freq = torch.fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = torch.fft.ifftn(x_freq, dim=(-2, -1)).real

    return x_filtered.to(x.dtype)


class FreeU(torch.nn.Module):
    def __init__(self, scale_map):
        super().__init__()
        self.scale_map = scale_map

    def forward(self, h, hsp, parameter, transformer_options):
        for k, scale in zip(self.scale_map, parameter):
            if k == h.shape[1]:
                h[:, : h.shape[1] // 2] = h[:, : h.shape[1] // 2] * scale[0]
                hsp = Fourier_filter(hsp, threshold=1, scale=scale[1])
        return h, hsp

    @staticmethod
    def from_closure(closure, transformer_options):
        scale_dict = {}
        for var_name, var in zip(closure.__code__.co_freevars, closure.__closure__):
            if var_name == "scale_dict":
                scale_dict = copy.deepcopy(var.cell_contents)
                break
        return FreeU(list(scale_dict.keys())), torch.Tensor(list(scale_dict.values()))

    def gen_cache_key(self):
        return [self.__class__.__name__, self.scale_map]


class FreeU_V2(torch.nn.Module):
    def __init__(self, scale_map):
        super().__init__()
        self.scale_map = scale_map

    def forward(self, h, hsp, parameter, transformer_options):
        for k, scale in zip(self.scale_map, parameter):
            if k == h.shape[1]:
                hidden_mean = h.mean(1).unsqueeze(1)
                B = hidden_mean.shape[0]
                hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True)
                hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
                hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (
                    hidden_max - hidden_min
                ).unsqueeze(2).unsqueeze(3)

                h[:, : h.shape[1] // 2] = h[:, : h.shape[1] // 2] * (
                    (scale[0] - 1) * hidden_mean + 1
                )

                hsp = Fourier_filter(hsp, threshold=1, scale=scale[1])

        return h, hsp

    @staticmethod
    def from_closure(closure, transformer_options):
        scale_dict = {}
        for var_name, var in zip(closure.__code__.co_freevars, closure.__closure__):
            if var_name == "scale_dict":
                scale_dict = copy.deepcopy(var.cell_contents)
                break
        return FreeU_V2(list(scale_dict.keys())), torch.Tensor(
            list(scale_dict.values())
        )

    def gen_cache_key(self):
        return [self.__class__.__name__, self.scale_map]
