import contextlib
import torch
import copy

from ..comfy_trace_utilities import ModuleFactory, hash_arg
from .nodes_freelunch import FreeU, FreeU_V2
from .nodes_model_downscale import (
    PatchModelAddDownscale_input_block_patch,
    PatchModelAddDownscale_output_block_patch,
)
from .openaimodel import PatchUNetModel

PATCH_PATCH_MAP = {
    "FreeU.patch.<locals>.output_block_patch": FreeU,
    "FreeU_V2.patch.<locals>.output_block_patch": FreeU_V2,
    "PatchModelAddDownscale.patch.<locals>.input_block_patch": PatchModelAddDownscale_input_block_patch,
    "PatchModelAddDownscale.patch.<locals>.output_block_patch": PatchModelAddDownscale_output_block_patch,
}


class BaseModelApplyModelModule(torch.nn.Module):
    def __init__(self, func, module):
        super().__init__()
        self.func = func
        self.module = module

    def forward(
        self,
        input_x,
        timestep,
        c_concat=None,
        c_crossattn=None,
        y=None,
        control=None,
        transformer_options={},
    ):
        kwargs = {"y": y}

        new_transformer_options = {}
        if "patches" in transformer_options:
            new_transformer_options["patches"] = transformer_options["patches"]

        return self.func(
            input_x,
            timestep,
            c_concat=c_concat,
            c_crossattn=c_crossattn,
            control=control,
            transformer_options=new_transformer_options,
            **kwargs,
        )


class BaseModelApplyModelModuleFactory(ModuleFactory):
    kwargs_name = (
        "input_x",
        "timestep",
        "c_concat",
        "c_crossattn",
        "y",
        "control",
    )

    def __init__(self, callable, kwargs) -> None:
        self.callable = callable
        self.unet_config = callable.__self__.model_config.unet_config
        self.kwargs = kwargs
        self.patch_module = {}
        self.patch_module_parameter = {}
        self.converted_kwargs = self.gen_converted_kwargs()

    def gen_converted_kwargs(self):
        converted_kwargs = {}
        for arg_name, arg in self.kwargs.items():
            if arg_name in self.kwargs_name:
                converted_kwargs[arg_name] = arg

        transformer_options = self.kwargs.get("transformer_options", {})
        patches = transformer_options.get("patches", {})

        patch_module = {}
        patch_module_parameter = {}

        for patch_type_name, patch_list in patches.items():
            patch_module[patch_type_name] = []
            patch_module_parameter[patch_type_name] = []
            for patch in patch_list:
                if patch.__qualname__ in PATCH_PATCH_MAP:
                    patch, parameter = PATCH_PATCH_MAP[patch.__qualname__].from_closure(
                        patch, transformer_options
                    )
                    patch_module[patch_type_name].append(patch)
                    patch_module_parameter[patch_type_name].append(parameter)
                    # output_block_patch_module.append(torch.jit.script(patch))
                else:
                    print(f"\33[93mWarning: Ignore patch {patch.__qualname__}.\33[0m")

        new_transformer_options = {}
        new_transformer_options["patches"] = patch_module_parameter
        if len(new_transformer_options["patches"]) > 0:
            converted_kwargs["transformer_options"] = new_transformer_options

        self.patch_module = patch_module
        self.patch_module_parameter = patch_module_parameter
        return converted_kwargs

    def gen_cache_key(self):
        key_kwargs = {}
        for k, v in self.converted_kwargs.items():
            if k == "transformer_options":
                nv = {}
                for tk, tv in v.items():
                    if not tk in ("patches"):  # ,"cond_or_uncond"
                        nv[tk] = tv
                v = nv
            key_kwargs[k] = v

        patch_module_cache_key = {}
        for patch_type_name, patch_list in self.patch_module.items():
            patch_module_cache_key[patch_type_name] = []
            for patch in patch_list:
                patch_module_cache_key[patch_type_name].append(patch.gen_cache_key())

        return (
            self.callable.__class__.__qualname__,
            hash_arg(self.unet_config),
            hash_arg(key_kwargs),
            hash_arg(patch_module_cache_key),
        )

    @contextlib.contextmanager
    def converted_module_context(self):
        module = BaseModelApplyModelModule(self.callable, self.callable.__self__)

        if len(self.patch_module) > 0:
            self.callable.__self__.diffusion_model = PatchUNetModel.cast_from(
                self.callable.__self__.diffusion_model
            )
            try:
                self.callable.__self__.diffusion_model.set_patch_module(
                    self.patch_module
                )

                yield (module, self.converted_kwargs)
            finally:
                self.callable.__self__.diffusion_model = (
                    self.callable.__self__.diffusion_model.cast_to_base_model()
                )
        else:
            yield (module, self.converted_kwargs)
