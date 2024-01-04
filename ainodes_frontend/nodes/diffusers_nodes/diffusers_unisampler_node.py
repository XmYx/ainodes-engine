import secrets
import subprocess

import torch
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline, StableDiffusionPipeline, \
    StableDiffusionImg2ImgPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLControlNetPipeline, \
    StableDiffusionXLInpaintPipeline

from ainodes_frontend.base.qimage_ops import pil2tensor, tensor2pil
# from ai_nodes.ainodes_engine_base_nodes.ainodes_backend import pil2tensor, torch_gc, tensor2pil
from backend_helpers.diffusers_helpers.diffusers_helpers import scheduler_type_values, SchedulerType, \
    get_scheduler
from backend_helpers.torch_helpers.torch_gc import torch_gc

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget

OP_NODE_DIFF_UNISAMPLER_DATA = get_next_opcode()
OP_NODE_DIFF_UNISAMPLER = get_next_opcode()

def dont_apply_watermark(images: torch.FloatTensor):

    return images

class DiffUniSamplerWidget(QDMNodeContentWidget):
    def initUI(self):
        self.create_check_box("Keep in VRAM", spawn="keepinvram")
        self.create_check_box("Tiled VAE", spawn="tiledvae")
        self.create_check_box("Attention Slicing", spawn="attentionslicing")
        self.create_check_box("Sequential Offload", spawn="modeloffload")
        self.create_main_layout(grid=1)

class DiffUniSamplerDataWidget(QDMNodeContentWidget):
    def initUI(self):

        #UNI
        self.create_combo_box(["latent", "pil"], "Return Type", spawn="return_type")
        self.scheduler_name = self.create_combo_box(scheduler_type_values, "Scheduler")
        self.prompt = self.create_text_edit("Linguistic Prompt", placeholder="Linguistic Prompt")
        self.prompt_2 = self.create_text_edit("Classic Prompt", placeholder="Classic Prompt")
        self.n_prompt = self.create_text_edit("Linguistic Negative Prompt", placeholder="Linguistic Negative Prompt")
        self.n_prompt_2 = self.create_text_edit("Classic Negative Prompt", placeholder="Classic Negative Prompt")
        self.height_val = self.create_spin_box("Height", min_val=64, max_val=4096, default_val=1024, step=64)
        self.width_val = self.create_spin_box("Width", min_val=64, max_val=4096, default_val=1024, step=64)
        self.steps = self.create_spin_box("Steps", min_val=1, max_val=4096, default_val=25, step=1)
        self.denoising_start = self.create_double_spin_box("Denoising Start", min_val=0, max_val=1.0, default_val=0.0, step=0.01)
        self.denoising_end = self.create_double_spin_box("Denoising End", min_val=0, max_val=1.0, default_val=1.0, step=0.01)
        self.scale = self.create_double_spin_box("Scale", min_val=0.01, max_val=25.00, default_val=7.5, step=0.01)
        self.eta = self.create_double_spin_box("Eta", min_val=0.00, max_val=1.00, default_val=1.0, step=0.01)
        self.seed = self.create_line_edit("Seed")
        self.top_crop = self.create_spin_box("Top Crop", min_val=0, max_val=4096, default_val=0, step=8)
        self.left_crop = self.create_spin_box("Left Crop", min_val=0, max_val=4096, default_val=0, step=8)


        #TXT2IMG ONLY
        self.guidance_rescale = self.create_double_spin_box("Guidance Rescale", min_val=0.00, max_val=25.00, default_val=0.0, step=0.01)
        self.original_width = self.create_spin_box("Orig Width", min_val=64, max_val=4096, default_val=1024, step=64)
        self.original_height = self.create_spin_box("Orig height", min_val=64, max_val=4096, default_val=1024, step=64)
        self.target_width = self.create_spin_box("Target width", min_val=64, max_val=4096, default_val=1024, step=64)
        self.target_height = self.create_spin_box("Target height", min_val=64, max_val=4096, default_val=1024, step=64)

        #IMG2IMG ONLY
        self.strength = self.create_double_spin_box("Strength", min_val=0.01, max_val=1.00, default_val=1.0, step=0.01)
        self.score = self.create_double_spin_box("Aesthetic score", min_val=0.01, max_val=25.00, default_val=6.0, step=0.01)
        self.n_score = self.create_double_spin_box("Negative Aesthetic score", min_val=0.01, max_val=25.00, default_val=2.5, step=0.01)

        self.create_main_layout(grid=2)

@register_node(OP_NODE_DIFF_UNISAMPLER_DATA)
class DiffSamplerDataNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
    help_text = "XL"
    op_code = OP_NODE_DIFF_UNISAMPLER_DATA
    op_title = "Diffusers UniSampler Data"
    content_label_objname = "sd_diff_sampler_data_node"
    category = "base/diffusers"
    NodeContent_class = DiffUniSamplerDataWidget
    dim = (500, 940)
    output_data_ports = [0]
    exec_port = 1
    def __init__(self, scene):
        super().__init__(scene, inputs=[5,5,2,6,1], outputs=[6,1])

    def evalImplementation_thread(self, index=0):

        mask = self.getInputData(0)

        if mask is not None:
            mask = tensor2pil(mask[0])

        latent = self.getInputData(2)
        prompt = self.content.prompt.toPlainText()
        prompt_2 = self.content.prompt_2.toPlainText()
        prompt_2 = prompt if prompt_2 == "" else prompt_2
        height = self.content.height_val.value()
        width = self.content.width_val.value()
        num_inference_steps = self.content.steps.value()
        denoising_start = self.content.denoising_start.value()
        denoising_end = self.content.denoising_end.value()
        guidance_scale = self.content.scale.value()
        negative_prompt = self.content.n_prompt.toPlainText()
        negative_prompt_2 = self.content.n_prompt_2.toPlainText()
        negative_prompt_2 = negative_prompt if negative_prompt_2 == "" else negative_prompt_2
        crops_coords_top_left = (self.content.top_crop.value(), self.content.left_crop.value())
        input_image = self.getInputData(1)
        image = latent
        if input_image is not None:
            image = tensor2pil(input_image[0])
        data = self.getInputData(3)
        cnet_scale = None
        start = None
        stop = None
        if data is not None:
            if "prompt" in data:
                prompt = data["prompt"]
            if "prompt_2" in data:
                prompt_2 = data["prompt_2"]
            if "negative_prompt" in data:
                negative_prompt = data["negative_prompt"]
            if "negative_prompt_2" in data:
                negative_prompt_2 = data["negative_prompt_2"]
            if "image" in data:
                image = data["image"]
            if "controlnet_conditioning_scale" in data:
                cnet_scale = data["controlnet_conditioning_scale"]
                start = data["control_guidance_start"]
                stop = data["control_guidance_end"]
        eta = self.content.eta.value()
        seed = secrets.randbelow(9999999999) if self.content.seed.text() == "" else int(self.content.seed.text())
        scheduler_name = self.content.scheduler_name.currentText()
        scheduler_enum = SchedulerType(scheduler_name)
        guidance_rescale = self.content.guidance_rescale.value()
        original_size = (self.content.original_width.value(), self.content.original_height.value())
        target_size = (self.content.target_width.value(), self.content.target_height.value())
        return_type = self.content.return_type.currentText()
        score = self.content.score.value()
        n_score = self.content.n_score.value()
        strength = self.content.strength.value()


        data = {
            "prompt":prompt,
            "prompt_2":prompt_2,
            "negative_prompt":negative_prompt,
            "negative_prompt_2":negative_prompt_2,
            "denoising_start":denoising_start,
            "denoising_end":denoising_end,
            "num_inference_steps":num_inference_steps,
            "eta":eta,
            "seed":seed,
            "scheduler":scheduler_enum,
            "guidance_rescale":guidance_rescale,
            "original_size":original_size,
            "target_size":target_size,
            "return_type":return_type,
            "aesthetic_score":score,
            "negative_aesthetic_score":n_score,
            "strength":strength,
            "height":height,
            "width":width,
            "guidance_scale":guidance_scale,
            "crops_coords_top_left":crops_coords_top_left,
            "image":image,
            "controlnet_conditioning_scale":cnet_scale,
            "control_guidance_start":start,
            "control_guidance_end":stop,
            "mask":mask,
        }

        return [data]
    def remove(self):
        super().remove()


@register_node(OP_NODE_DIFF_UNISAMPLER)
class DiffSamplerNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
    help_text = "XL"
    op_code = OP_NODE_DIFF_UNISAMPLER
    op_title = "Diffusers UniSampler"
    content_label_objname = "sd_diff_sampler_node"
    category = "base/diffusers"
    NodeContent_class = DiffUniSamplerWidget
    dim = (340, 340)
    output_data_ports = [0, 1, 2]
    exec_port = 1
    use_gpu = True
    make_dirty = True
    def __init__(self, scene):
        super().__init__(scene, inputs=[4,6,1], outputs=[5,2,6,1])


    def evalImplementation_thread(self, index=0):

        pipe = self.getInputData(0)
        data = self.getInputData(1)

        gpu_id = self.content.gpu_id.currentText()
        from ainodes_frontend import singleton as gs
        keepinvram = self.content.keepinvram.isChecked()
        tiledvae = self.content.tiledvae.isChecked()
        attentionslicing = self.content.attentionslicing.isChecked()
        modeloffload = self.content.modeloffload.isChecked()

        if tiledvae:
            try:
                pipe.enable_vae_slicing()
                pipe.enable_vae_tiling()
            except:
                pass
        else:
            try:
                pipe.disable_vae_slicing()
                pipe.disable_vae_tiling()
            except:
                pass

        if attentionslicing:
            try:
                pipe.enable_attention_slicing()
            except:
                pass
        else:
            try:
                pipe.disable_attention_slicing()
            except:
                pass

        if modeloffload:
            try:
                pipe.enable_model_cpu_offload()
            except:
                pass
        else:
            try:
                pipe.disable_model_cpu_offload()
            except:
                pass

        get_scheduler(pipe, data["scheduler"])

        target_device = f"{gs.device.type}:{gpu_id}"

        if pipe.device != target_device and not modeloffload:
            pipe.to(f"{gs.device.type}:{gpu_id}")

        seed = int(data["seed"])
        generator = torch.Generator(target_device).manual_seed(seed)
        latents = None

        from backend_helpers.torch_helpers.rng_noise_generator import ImageRNGNoise
        rng = ImageRNGNoise((4, data["height"] // 8, data["width"] // 8), [data["seed"]], [data["seed"] - 1], 0.6, 1024, 1024 )
        noise = rng.first().half()
        # noise = noise.unsqueeze(0)
        #print(noise.shape)
        args = {
            "prompt": data["prompt"],
            "negative_prompt": data["negative_prompt"],
            "num_inference_steps": data["num_inference_steps"],
            "eta": data["eta"],
            "guidance_scale": data["guidance_scale"],
            "generator":generator,
            "latents":noise
        }

        if isinstance(pipe, StableDiffusionImg2ImgPipeline) or isinstance(pipe, StableDiffusionXLImg2ImgPipeline):
            args["image"] = data["image"]
            args["strength"] = data["strength"]

        if isinstance(pipe, StableDiffusionXLImg2ImgPipeline) or isinstance(pipe, StableDiffusionXLPipeline) or isinstance(pipe, StableDiffusionXLInpaintPipeline):
            args["prompt_2"] = data["prompt_2"]
            args["negative_prompt_2"] = data["negative_prompt_2"]
            args["denoising_end"] = data["denoising_end"]
            args["original_size"] = data["original_size"]
            args["crops_coords_top_left"] = data["crops_coords_top_left"]
            args["target_size"] = data["target_size"]

        if isinstance(pipe, StableDiffusionXLImg2ImgPipeline):
            args["aesthetic_score"] = data["aesthetic_score"]
            args["negative_aesthetic_score"] = data["negative_aesthetic_score"]
            args["denoising_start"] = data["denoising_start"]

        if isinstance(pipe, StableDiffusionXLControlNetPipeline):
            args["image"] = data["image"]
            args["controlnet_conditioning_scale"] = data["controlnet_conditioning_scale"][0]
            args["guess_mode"] = False
            args["control_guidance_start"] = data["control_guidance_start"][0]
            args["control_guidance_end"] = data["control_guidance_end"][0]

        if isinstance(pipe, StableDiffusionXLInpaintPipeline):
            args["image"] = data["image"]
            args["mask_image"] = data["mask"]
        if isinstance(pipe, StableDiffusionXLPipeline):
            args["width"] = data["width"]
            args["height"] = data["height"]
            latents = None

            #latents, image = pipe.generate(**args)
            image = pipe(**args).images[0]
            # image = image[0]
        else:

            print(f"[ UNISAMPLER GENERATING WITH ARGS: {args} ]")
            image = pipe(**args).images[0]

        if not keepinvram:
            pipe.to("cpu")

        return [pil2tensor(image), latents, args]

    def remove(self):
        super().remove()
