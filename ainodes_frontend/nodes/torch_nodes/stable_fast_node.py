import torch
from sfast.compilers.diffusion_pipeline_compiler import CompilationConfig

from backend_helpers.sfast_helpers.sfast_compiler import build_lazy_trace_module
import os
from qtpy import QtCore, QtGui
from qtpy import QtWidgets

# from ..ainodes_backend.model_loader import ModelLoader
# from ..ainodes_backend import torch_gc

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget


from ainodes_frontend import singleton as gs
from backend_helpers.torch_helpers.model_loader import ModelLoader
from backend_helpers.torch_helpers.torch_gc import torch_gc

# from ..ainodes_backend.sd_optimizations.sd_hijack import valid_optimizations

OP_NODE_SFAST = get_next_opcode()

def is_cuda_malloc_async():
    return "cudaMallocAsync" in torch.cuda.get_allocator_backend()


def gen_stable_fast_config():
    config = CompilationConfig.Default()
    # xformers and triton are suggested for achieving best performance.
    # It might be slow for triton to generate, compile and fine-tune kernels.
    try:
        import xformers

        config.enable_xformers = True
    except ImportError:
        print("xformers not installed, skip")
    try:
        import triton

        config.enable_triton = True
    except ImportError:
        print("triton not installed, skip")

    if config.enable_triton and is_cuda_malloc_async():
        print("disable stable fast triton because of cudaMallocAsync")
        config.enable_triton = False

    # CUDA Graph is suggested for small batch sizes.
    # After capturing, the model only accepts one fixed image size.
    # If you want the model to be dynamic, don't enable it.
    config.enable_cuda_graph = True
    # config.enable_jit_freeze = False
    return config


class StableFastPatch:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.stable_fast_model = None

    def __deepcopy__(self, memo=None):
        return self

    def __call__(self, model_function, params):
        input_x = params.get("input")
        timestep_ = params.get("timestep")
        c = params.get("c")

        # disable with accelerate for now
        if hasattr(model_function.__self__, "hf_device_map"):
            return model_function(input_x, timestep_, **c)

        if self.stable_fast_model is None:
            self.stable_fast_model = build_lazy_trace_module(
                self.config,
                input_x.device,
                id(self),
            )

        return self.stable_fast_model(
            model_function, input_x=input_x, timestep=timestep_, **c
        )

    def to(self, device):
        if type(device) == torch.device:
            if self.config.enable_cuda_graph or self.config.enable_jit_freeze:
                if device.type == "cpu":
                    # comfyui tell we should move to cpu. but we cannt do it with cuda graph and freeze now.
                    del self.stable_fast_model
                    self.stable_fast_model = None
                    print(
                        "\33[93mWarning: Your graphics card doesn't have enough video memory to keep the model. If you experience a noticeable delay every time you start sampling, please consider disable enable_cuda_graph.\33[0m"
                    )
            else:
                if self.stable_fast_model != None and device.type == "cpu":
                    self.stable_fast_model.to_empty()
        return self


class SFastWidget(QDMNodeContentWidget):
    def initUI(self):
        self.create_main_layout(grid=1)

@register_node(OP_NODE_SFAST)
class StableFastNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/torch.png"
    op_code = OP_NODE_SFAST
    op_title = "Stable-Fast Apply"
    content_label_objname = "sfast_node"
    category = "base/loaders"
    input_socket_name = ["MODEL", "EXEC"]
    # output_socket_name = ["EXEC"]
    custom_output_socket_name = ["MODEL", "EXEC"]

    NodeContent_class = SFastWidget
    dim = (340, 340)
    def __init__(self, scene):
        super().__init__(scene, inputs=[4,1], outputs=[4,1])

    def evalImplementation_thread(self, index=0):

        model = self.getInputData(0)

        if not hasattr(model, "stable_fast_patch"):
            model.stable_fast_patch = True
            config = gen_stable_fast_config()
            if config.memory_format is not None:
                model.model.to(memory_format=config.memory_format)
            patch = StableFastPatch(model, config)
            model.set_model_unet_function_wrapper(patch)
        return [model]