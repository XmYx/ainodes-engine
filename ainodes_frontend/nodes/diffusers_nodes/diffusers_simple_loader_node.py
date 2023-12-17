import torch
import os

from backend_helpers.diffusers_helpers.diffusers_helpers import diffusers_models, diffusers_indexed, \
    scheduler_type_values, SchedulerType, get_scheduler_class
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget

OP_NODE_DIFF_SIMPLE_PIPE = get_next_opcode()
from diffusers import (StableDiffusionPipeline, StableDiffusionXLPipeline,
                       StableDiffusionImg2ImgPipeline, StableDiffusionControlNetPipeline,
                       StableDiffusionXLImg2ImgPipeline, StableDiffusionControlNetImg2ImgPipeline,
                       StableDiffusionXLControlNetPipeline, StableDiffusionXLInpaintPipeline)
from ainodes_frontend import singleton as gs

class DiffSDSimplePipelineWidget(QDMNodeContentWidget):
    def initUI(self):

        self.create_combo_box([item["name"] for item in diffusers_models], "Model", spawn="models")
        self.create_check_box("XL", spawn="xl")
        self.create_check_box("TinyVAE", spawn="tinyvae")

        self.create_check_box("Use Local Model", spawn="use_local_models")
        checkpoint_folder = gs.prefs.checkpoints
        os.makedirs(checkpoint_folder, exist_ok=True)
        checkpoint_files = [f for f in os.listdir(checkpoint_folder) if f.endswith(('.ckpt', '.pt', '.bin', '.pth', '.safetensors'))]
        self.local_model = self.create_combo_box(checkpoint_files, "Model:")
        if checkpoint_files == []:
            self.local_model.addItem("Please place a model in models/checkpoints_xl")
            print(f"TORCH LOADER NODE: No model file found at {os.getcwd()}/models/checkpoints,")
            print(f"TORCH LOADER NODE: please download your favorite ckpt before Evaluating this node.")


        self.create_combo_box(["txt2img", "img2img", "txt2img_cnet", "img2img_cnet", "txt2img_xl", "img2img_xl", "txt2img_xl_cnet","img2img_xl_cnet",], "pipeline", spawn="pipe_select")
        self.create_main_layout(grid=1)

@register_node(OP_NODE_DIFF_SIMPLE_PIPE)
class DiffSDPipelineNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
    help_text = "Diffusers SDPipeline"
    op_code = OP_NODE_DIFF_SIMPLE_PIPE
    op_title = "Diffusers - Simple Pipeline"
    content_label_objname = "diff_simple_pipeline_node"
    category = "base/diffusers"
    NodeContent_class = DiffSDSimplePipelineWidget
    dim = (340, 460)
    output_data_ports = [0]

    custom_input_socket_name = ["CONTROLNET", "EXEC"]

    def __init__(self, scene):
        super().__init__(scene, inputs=[4,1], outputs=[4,1])
        self.pipe = None
    def evalImplementation_thread(self, index=0):


        isxl = self.content.xl.isChecked()
        controlnet = self.getInputData(0)

        # scheduler_name = self.content.schedulers.currentText()
        # scheduler = SchedulerType(scheduler_name)
        # scheduler_class = get_scheduler_class(scheduler)
        model_key = self.content.models.currentIndex()
        model_name = diffusers_indexed[model_key]
        # scheduler = scheduler_class.from_pretrained(model_name, subfolder="scheduler")

        pipe_select = self.content.pipe_select.currentText()

        pipes = {"txt2img_xl":StableDiffusionXLPipeline,
                 "img2img_xl":StableDiffusionXLImg2ImgPipeline,
                 "txt2img_cnet":StableDiffusionControlNetPipeline,
                 "img2img_cnet":StableDiffusionControlNetImg2ImgPipeline,
                 "txt2img_xl_cnet":StableDiffusionXLControlNetPipeline,
                 "img2img_xl_cnet":StableDiffusionControlNetImg2ImgPipeline}



        pipe_class = pipes.get(pipe_select, StableDiffusionXLPipeline)

        args = {"torch_dtype":torch.float16,
                "use_auth_token":"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"}

        if "cnet" in pipe_select:
            args["controlnet"] = controlnet

        if not self.content.use_local_models.isChecked():
            args["pretrained_model_name_or_path"] = model_name
            self.pipe = pipe_class.from_pretrained(**args)
        else:
            model_name = f"{gs.prefs.checkpoints}/{self.content.local_model.currentText()}"

            args["pretrained_model_link_or_path"] = model_name

            self.pipe = pipe_class.from_single_file(**args)

        if isinstance(self.pipe, StableDiffusionXLPipeline):

            print("Adding custom Generate call to Diffusers Stable Diffusion XL Pipeline")

            from backend_helpers.diffusers_helpers.diffusers_xl_call import new_call

            def replace_call(pipe, new_call):
                def call_with_self(*args, **kwargs):
                    return new_call(pipe, *args, **kwargs)

                return call_with_self

            self.pipe.generate = replace_call(self.pipe, new_call)

        tinyvae = self.content.tinyvae.isChecked()


        from sfast.compilers.stable_diffusion_pipeline_compiler import (
            compile, CompilationConfig)

        config = CompilationConfig.Default()
        # xformers and Triton are suggested for achieving best performance.
        try:
            import xformers
            config.enable_xformers = True
        except ImportError:
            print('xformers not installed, skip')
        try:
            import triton
            config.enable_triton = True
        except ImportError:
            print('Triton not installed, skip')
        # CUDA Graph is suggested for small batch sizes and small resolutions to reduce CPU overhead.
        config.enable_cuda_graph = True
        self.pipe = compile(self.pipe, config)
        if tinyvae:
            tiny_model = "madebyollin/taesdxl"
            from diffusers import AutoencoderTiny
            self.pipe.vae = AutoencoderTiny.from_pretrained(tiny_model, torch_dtype=torch.float16)

        #self.pipe.unet = torch.compile(self.pipe.unet, fullgraph=True, mode="max-autotune")

        return [self.pipe]
    def remove(self):
        try:
            if self.pipe != None:
                self.pipe.to("cpu")
                del self.pipe
                self.pipe = None
        except:
            self.pipe = None
        super().remove()