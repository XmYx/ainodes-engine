import io
import os
import numpy as np
import torch
from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import pyqtSignal, QUrl
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QFileDialog, QLabel, QVBoxLayout

from node_core.node_register import tensor_image_to_pixmap, AiNode, register_node
from nodeeditor.node_content_widget import QDMNodeContentWidget

from modules.conditioner import HFEmbedder
from modules.autoencoder import AutoEncoder, AutoEncoderParams
from modules.flux_model import Flux, FluxParams
from util import ModelSpec
from flux_pipeline import FluxPipeline

from safetensors.torch import load_file as load_sft
class T5LoaderWidget(QDMNodeContentWidget):
    def initUI(self):
        self.create_main_layout(grid=1)


#TODO OVERRIDE THIS MONSTROSITY

base_config = {
  "version": "flux-dev",
  "params": {
    "in_channels": 64,
    "vec_in_dim": 768,
    "context_in_dim": 4096,
    "hidden_size": 3072,
    "mlp_ratio": 4.0,
    "num_heads": 24,
    "depth": 19,
    "depth_single_blocks": 38,
    "axes_dim": [
      16,
      56,
      56
    ],
    "theta": 10000,
    "qkv_bias": True,
    "guidance_embed": True
  },
  "ae_params": {
    "resolution": 256,
    "in_channels": 3,
    "ch": 128,
    "out_ch": 3,
    "ch_mult": [
      1,
      2,
      4,
      4
    ],
    "num_res_blocks": 2,
    "z_channels": 16,
    "scale_factor": 0.3611,
    "shift_factor": 0.1159
  },
  "ckpt_path": "/home/mix/Playground/ComfyUI/models/unet/flux1-dev.safetensors",
  "ae_path": "/home/mix/Playground/ComfyUI/models/vae/flux-ae.safetensors",
  "repo_id": "black-forest-labs/FLUX.1-dev",
  "repo_flow": "flux1-dev.sft",
  "repo_ae": "ae.sft",
  "text_enc_max_length": 512,
  "text_enc_path": "city96/t5-v1_1-xxl-encoder-bf16",
  "text_enc_device": "cuda:0",
  "ae_device": "cuda:0",
  "flux_device": "cuda:0",
  "flow_dtype": "bfloat16",
  "ae_dtype": "bfloat16",
  "text_enc_dtype": "bfloat16",
  "text_enc_quantization_dtype": "qfloat8",
  "ae_quantization_dtype": None,
  "compile_extras": True,
  "compile_blocks": True,
  "offload_ae": False,
  "offload_text_encoder": False,
  "offload_flow": False
}

main_spec = ModelSpec(**base_config)

@register_node()
class T5LoaderNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/input_image.png"
    op_code = 0
    op_title = "Load T5"
    content_label_objname = "t5_loader_node"
    category = "base/image"
    custom_output_socket_names = ["T5", "EXEC"]
    dim = (300, 120)
    NodeContent_class = T5LoaderWidget
    # Class-level variable to cache the loaded model

    def __init__(self, scene):
        super().__init__(scene, inputs=[1], outputs=[9, 1])
        self._t5_model_cache = None

    def evalImplementation_thread(self, index=0):
        # Check if the model is already loaded
        if self._t5_model_cache is None:
            with torch.inference_mode():

                # Load the model and store it in the cache
                self._t5_model_cache = HFEmbedder(
                    'city96/t5-v1_1-xxl-encoder-bf16',
                    max_length=512,
                    torch_dtype=torch.bfloat16,
                    device=torch.device('cuda'),
                    quantization_dtype='qfloat8',
                ).to(torch.bfloat16)
                print("Loaded T5")

        # Return the cached model
        return [self._t5_model_cache]


@register_node()
class ClipLoaderNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/input_image.png"
    op_code = 0
    op_title = "Load Clip Flux"
    content_label_objname = "clip_loader_node"
    category = "base/image"
    custom_output_socket_names = ["CLIP", "EXEC"]
    dim = (300, 120)
    NodeContent_class = T5LoaderWidget
    # Class-level variable to cache the loaded model

    def __init__(self, scene):
        super().__init__(scene, inputs=[1], outputs=[9, 1])
        self._clip_model_cache = None

    def evalImplementation_thread(self, index=0):
        # Check if the model is already loaded
        if self._clip_model_cache is None:
            with torch.inference_mode():

                # Load the model and store it in the cache
                self._clip_model_cache = HFEmbedder(
                    'openai/clip-vit-large-patch14',
                    max_length=77,
                    torch_dtype=torch.bfloat16,
                    device=torch.device('cuda'),
                    quantization_dtype='qfloat8',
                ).to(torch.bfloat16)
                print("Loaded CLIP")


        # Return the cached model
        return [self._clip_model_cache]


@register_node()
class AELoaderNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/input_image.png"
    op_code = 0
    op_title = "Load VAE"
    content_label_objname = "vae_loader_node"
    category = "base/image"
    custom_output_socket_names = ["VAE", "EXEC"]
    dim = (300, 120)
    NodeContent_class = T5LoaderWidget
    # Class-level variable to cache the loaded model

    def __init__(self, scene):
        super().__init__(scene,  inputs=[1], outputs=[9, 1])
        self._vae_model_cache = None

    def evalImplementation_thread(self, index=0):
        # Check if the model is already loaded
        if self._vae_model_cache is None:
            with torch.inference_mode():

                # Load the model and store it in the cache
                params = AutoEncoderParams(**ae_params)
                self._vae_model_cache = AutoEncoder(params).to(torch.bfloat16)
                #TODO ADD CKPT PATH
                ckpt_path = "/home/mix/Playground/ComfyUI/models/vae/flux-ae.safetensors"
                sd = load_sft(ckpt_path, device='cpu')
                missing, unexpected = self._vae_model_cache.load_state_dict(sd, strict=False, assign=True)
                self._vae_model_cache.to(device='cuda', dtype=torch.bfloat16)
                from float8_quantize import recursive_swap_linears
                recursive_swap_linears(self._vae_model_cache)
                print("Loaded VAE")
                del missing, unexpected, sd
                torch.cuda.empty_cache()
        # Return the cached model
        return [self._vae_model_cache]


@register_node()
class TransformerLoaderNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/input_image.png"
    op_code = 0
    op_title = "Load Flux"
    content_label_objname = "transformer_loader_node"
    category = "base/image"
    custom_output_socket_names = ["FLUX", "EXEC"]
    dim = (300, 120)
    NodeContent_class = T5LoaderWidget
    # Class-level variable to cache the loaded model


    def __init__(self, scene):
        super().__init__(scene, inputs=[1], outputs=[9, 1])
        self._flux_model_cache = None
    def evalImplementation_thread(self, index=0):
        # Check if the model is already loaded
        if self._flux_model_cache is None:
            # Load the model and store it in the cache
            with torch.inference_mode():

                self._flux_model_cache = Flux(main_spec, dtype=torch.bfloat16)
                sd = load_sft(main_spec.ckpt_path, device="cpu")
                missing, unexpected = self._flux_model_cache.load_state_dict(sd, strict=False, assign=True)
                self._flux_model_cache.type(torch.bfloat16)
                from float8_quantize import quantize_flow_transformer_and_dispatch_float8
                self._flux_model_cache = quantize_flow_transformer_and_dispatch_float8(
                    self._flux_model_cache,
                    'cuda',
                    offload_flow=True,
                    swap_linears_with_cublaslinear=False,
                    flow_dtype=torch.bfloat16,
                    quantize_modulation=False,
                    quantize_flow_embedder_layers=True,
                )
                print("Loaded FLUX")
                del missing, unexpected, sd
                torch.cuda.empty_cache()

        # Return the cached model
        return [self._flux_model_cache]

ae_params = {
    "resolution": 256,
    "in_channels": 3,
    "ch": 128,
    "out_ch": 3,
    "ch_mult": [
      1,
      2,
      4,
      4
    ],
    "num_res_blocks": 2,
    "z_channels": 16,
    "scale_factor": 0.3611,
    "shift_factor": 0.1159
  }


class FluxSamplerWidget(QDMNodeContentWidget):

    set_image_signal = pyqtSignal(QPixmap)

    def initUI(self):
        self.create_text_edit("Prompt", placeholder="Linguistic Prompt (XL)", spawn='prompt')

        self.steps = self.create_spin_box("Steps:", 1, 10000, 10)
        self.width_value = self.create_spin_box("Width", 64, 4096, 512, 64)
        self.height_value = self.create_spin_box("Height", 64, 4096, 512, 64)
        self.create_combo_box('Scheduler', [
            'original',
            'simple',
            'ddim',
            'normal',
            'beta',
            'cosine',
            'exponential',
            'linear',
            'quadratic',
            'logarithmic',
            # 'sigmoid',
            'sinusoidal',
            'piecewise_linear',
            'random_walk',
            'inverse_square_root',
            'tanh',
            'custom'
        ], spawn='scheduler')

        self.create_combo_box('Sampler', [
            'original',
            'euler',
            'euler_ancestras',
            'heun',
            'dpm_2',
            'dpm_2_ancestral',
            'lms',
        ], spawn='sampler')
        self.create_line_edit('seed', spawn='seed')
        self.create_check_box('Offload Flow', spawn='offload_flow')
        self.create_check_box('Offload TE', spawn='offload_te')
        self.create_check_box('Offload VAE', spawn='offload_vae')
        self.create_check_box('Compile Blocks', spawn='compile_blocks')
        self.create_check_box('Compile Extras', spawn='compile_extras')



        self.create_main_layout(grid=1)

@register_node()
class FluxPipelineNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/input_image.png"
    op_code = 0
    op_title = "Flux Pipeline"
    content_label_objname = "flux_pipeline_node"
    category = "base/image"
    custom_input_socket_names = ["CLIP", "T5", "FLUX", "VAE"]
    custom_output_socket_names = ["IMAGE", "EXEC"]
    dim = (300, 700)
    NodeContent_class = FluxSamplerWidget
    # Class-level variable to cache the loaded model

    def __init__(self, scene):
        super().__init__(scene, inputs=[9,9,9,9,1], outputs=[5, 1])
        self._flux_pipeline_cache = None

    def evalImplementation_thread(self, index=0):
        # Check if the model is already loaded
        if self._flux_pipeline_cache is None:
            with torch.inference_mode():
                self._flux_pipeline_cache = FluxPipeline(name='flux-dev',
                        offload= self.content.offload_flow.isChecked(),
                        clip = self.getInputData(0),
                        t5 = self.getInputData(1),
                        model = self.getInputData(2),
                        ae = self.getInputData(3),
                        dtype = torch.bfloat16,
                        verbose = False,
                        flux_device = "cuda:0",
                        ae_device  = "cuda:0",
                        clip_device  = "cuda:0",
                        t5_device = "cuda:0",
                        config = main_spec,
                        debug = False)

        self._flux_pipeline_cache.offload_flow = self.content.offload_flow.isChecked()
        self._flux_pipeline_cache.offload_text_encoder = self.content.offload_te.isChecked()
        self._flux_pipeline_cache.offload_vae = self.content.offload_vae.isChecked()

        self._flux_pipeline_cache.config.compile_blocks = self.content.compile_blocks.isChecked()
        self._flux_pipeline_cache.config.compile_extras = self.content.compile_extras.isChecked()
        if self._flux_pipeline_cache.config.compile_blocks or self._flux_pipeline_cache.config.compile_extras:
            self._flux_pipeline_cache.compile(self.getInputData(0),self.getInputData(1),self.getInputData(2),self.getInputData(3))


        with torch.inference_mode():
            image_io_bytes =  self._flux_pipeline_cache.generate(
                        self.getInputData(0),
                        self.getInputData(1),
                        self.getInputData(2),
                        self.getInputData(3),
                        prompt=self.content.prompt.toPlainText(),
                        width = self.content.width_value.value(),
                        height = self.content.height_value.value(),
                        num_steps = self.content.steps.value(),
                        guidance = 3.5,
                        seed = self.content.seed.text(),
                        init_image = None,
                        strength = 1.0,
                        silent = False,
                        num_images = 1,
                        return_seed = False,
                        jpeg_quality = 100,
                        scheduler_type=self.content.scheduler.currentText(),
                        sampler_type=self.content.sampler.currentText())
        # Return the cached model
        return [image_io_bytes]


class ImagePreviewWidget(QDMNodeContentWidget):
    set_image_signal = pyqtSignal(QPixmap)

    def initUI(self):
        # Create label to display the image
        self.image_label = QLabel()
        self.image_label.setScaledContents(False)  # Don't scale the image
        self.image_label.setPixmap(QPixmap())  # Start with an empty pixmap

        # Create a checkbox for saving the output
        self.create_check_box('Save Output', checked=False, spawn='save_output', object_name='save_output')

        # Set up the layout
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.save_output)
        self.setLayout(layout)

        # Connect the signal
        self.set_image_signal.connect(self.image_label.setPixmap)

from node_core.node_register import register_node, AiNode

@register_node()
class ImagePreviewNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/input_image.png"
    op_code = 0
    op_title = "Image Preview"
    content_label_objname = "image_preview_node"
    category = "base/image"
    custom_input_socket_names = ["IMAGE", "EXEC"]
    dim = (300, 120)
    NodeContent_class = ImagePreviewWidget

    def __init__(self, scene):
        super().__init__(scene, inputs=[5,1], outputs=[1])

    def evalImplementation_thread(self, index=0):
        image_input = self.getInputData(0)
        if image_input is not None:
            # Handle different types of input
            if isinstance(image_input, dict):
                # If image_input is a dict, try to get 'samples'
                image_tensor = image_input.get('samples', None)
            else:
                image_tensor = image_input
            if image_tensor is not None:
                # Assuming image_tensor is a batched tensor of shape [batch_size, C, H, W] or [batch_size, H, W, C]
                if len(image_tensor.shape) == 4:
                    # Batch dimension present
                    image_tensor = image_tensor[0]  # Take first image in batch
                # Now image_tensor should be of shape [C, H, W] or [H, W, C]
                if image_tensor.shape[0] == 3 or image_tensor.shape[0] == 4:
                    # Shape is [C, H, W], transpose to [H, W, C]
                    image_np = image_tensor.cpu().numpy()
                    image_np = np.transpose(image_np, (1, 2, 0))
                else:
                    # Shape is [H, W, C]
                    image_np = image_tensor.cpu().numpy()
                # Convert from float [0,1] to uint8 [0,255]
                image_np = (image_np * 255).clip(0, 255).astype(np.uint8)
                # Handle possible alpha channel
                if image_np.shape[2] == 4:
                    # Has alpha channel
                    image_pil = Image.fromarray(image_np, mode='RGBA')
                else:
                    image_pil = Image.fromarray(image_np)
                # Convert to QPixmap
                pixmap = QPixmap.fromImage(ImageQt(image_pil))
                self.resize(pixmap)
                self.content.set_image_signal.emit(pixmap)
                # If save output is checked, save the image
                if self.content.save_output.isChecked():
                    # Save the image
                    output_path = 'output_image.png'  # Customize the path or filename as needed
                    image_pil.save(output_path)
                    print(f"Image saved to {output_path}")

    def resize(self, pixmap):
        self.grNode.setToolTip("")
        dims = [pixmap.size().height() + 360, pixmap.size().width() + 30]
        if self.dim != dims:
            self.dim = dims
            self.grNode.height = dims[0]
            self.grNode.width = dims[1]
            self.content.setGeometry(0, 25, pixmap.size().width() + 32, pixmap.size().height() + 250)
            self.update_all_sockets()

