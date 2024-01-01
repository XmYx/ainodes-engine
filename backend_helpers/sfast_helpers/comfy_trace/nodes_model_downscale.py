import torch
import comfy.utils


class PatchModelAddDownscale_input_block_patch(torch.nn.Module):
    def __init__(
        self,
        block_number,
        downscale_method,
        downscale_factor,
        sigma,
        sigma_start,
        sigma_end,
    ):
        super().__init__()
        self.block_number = block_number
        self.downscale_method = downscale_method
        self.downscale_factor = downscale_factor
        self.sigma = sigma
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end

    def forward(self, h, parameter, transformer_options):
        if transformer_options["block"][1] == self.block_number:
            if self.sigma <= self.sigma_start and self.sigma >= self.sigma_end:
                h = comfy.utils.common_upscale(
                    h,
                    round(int(h.shape[-1]) * (1.0 / self.downscale_factor)),
                    round(int(h.shape[-2]) * (1.0 / self.downscale_factor)),
                    self.downscale_method,
                    "disabled",
                )
        return h

    @staticmethod
    def from_closure(closure, transformer_options):
        parameter_dict = {}
        for var_name, var in zip(closure.__code__.co_freevars, closure.__closure__):
            parameter_dict[var_name] = var.cell_contents

        sigma = transformer_options["sigmas"][0].item()
        return (
            PatchModelAddDownscale_input_block_patch(
                parameter_dict["block_number"],
                parameter_dict["downscale_method"],
                parameter_dict["downscale_factor"],
                sigma,
                parameter_dict["sigma_start"],
                parameter_dict["sigma_end"],
            ),
            (),
        )

    def gen_cache_key(self):
        flag = 0
        if self.sigma <= self.sigma_start and self.sigma >= self.sigma_end:
            flag = 1
        return [
            self.__class__.__name__,
            flag,
            self.block_number,
            self.downscale_method,
            self.downscale_factor,
        ]


class PatchModelAddDownscale_output_block_patch(torch.nn.Module):
    def __init__(self, upscale_method):
        super().__init__()
        self.upscale_method = upscale_method

    def forward(self, h, hsp, parameter, transformer_options):
        if h.shape[2] != hsp.shape[2]:
            h = comfy.utils.common_upscale(
                h,
                int(hsp.shape[-1]),
                int(hsp.shape[-2]),
                self.upscale_method,
                "disabled",
            )
        return h, hsp

    @staticmethod
    def from_closure(closure, transformer_options):
        parameter_dict = {}
        for var_name, var in zip(closure.__code__.co_freevars, closure.__closure__):
            parameter_dict[var_name] = var.cell_contents
        return (
            PatchModelAddDownscale_output_block_patch(parameter_dict["upscale_method"]),
            (),
        )

    def gen_cache_key(self):
        return [self.__class__.__name__, self.upscale_method]
