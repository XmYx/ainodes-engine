import json
import threading
import time

from PIL import Image

# from modules.flux_core import encode, Guider_Basic, get_sigmas, Noise_RandomNoise, sample_fn, model_manager
# from modules.sampling import samplers
#
# import torch
# import secrets
#
# # from node_core.register import register_node, NODE_REGISTRY
#
#
# import threading
import time
import os
import secrets
# import torch
# import onnx
# import onnxruntime as ort
# from onnxruntime.quantization import quantize_dynamic, QuantType

from modules.flux_core import encode, Guider_Basic, get_sigmas, Noise_RandomNoise, sample_fn, model_manager
from modules.sampling import samplers
import torch
import onnxruntime as ort
import numpy as np
import torch.nn as nn

def postprocess(tensor):
    tensor = (tensor * 0.5 + 0.5) * 255
    tensor = tensor.permute(1, 2, 0).clamp(0, 255).to(torch.uint8).cpu().numpy()
    return Image.fromarray(tensor)


import torch
import torch.jit


import torch
import json
import torch.nn as nn

class CachedModule(nn.Module):
    def __init__(self, original_layer, layer_name, cache):
        super(CachedModule, self).__init__()
        self.original_layer = original_layer
        self.layer_name = layer_name
        self.cache = cache

    def forward(self, *args, **kwargs):
        if self.layer_name in self.cache:
            return self.cache[self.layer_name]

        output = self.original_layer(*args, **kwargs)
        self.cache[self.layer_name] = output
        return output

def apply_caching_and_compilation(model, layers_info_file="layers_info.json"):
    with open(layers_info_file, "r") as f:
        layers_info = json.load(f)

    cache = {}

    def replace_layer_with_cached(layer_name, layer_func):
        """
        Replaces a layer in the model with a cached version based on similarity.
        """
        similarity = layers_info[layer_name].get("similarity_to_previous", None)

        if similarity == 1.0:
            return CachedModule(layer_func, layer_name, cache)
        else:
            return layer_func

    for layer_name, _ in layers_info.items():
        layer_func = getattr(model, layer_name, None)
        if layer_func:
            new_layer_func = replace_layer_with_cached(layer_name, layer_func)
            setattr(model, layer_name, new_layer_func)

    return model
class DynamicAutoencoder(nn.Module):
    def __init__(self, latent_scale, latent_channels, in_blocks, out_blocks):
        super(DynamicAutoencoder, self).__init__()

        self.latent_scale = latent_scale
        self.latent_channels = latent_channels

        # Encoder
        encoder_layers = []
        in_channels = 3  # Assuming input images have 3 channels (RGB)
        for i, out_channels in enumerate(in_blocks):
            if i < 3:  # Apply downsampling only in the first 3 layers
                encoder_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1))
            else:
                encoder_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            encoder_layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        self.encoder = nn.Sequential(*encoder_layers)

        # Adjust the latent_conv layer to ensure it matches the latent_channels
        self.latent_conv = nn.Conv2d(in_channels, latent_channels, kernel_size=1)

        # Decoder
        decoder_layers = []
        in_channels = latent_channels
        for i, out_channels in enumerate(out_blocks):
            # Upsample first, then apply the convolution for better control
            if i < 3:  # Upscale in the first 3 layers to match the downscaling in the encoder
                decoder_layers.append(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
                )
            else:  # If no more upscaling is needed, just apply regular convolution
                decoder_layers.append(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
                )
            decoder_layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        # Final layer to output 3 channels (RGB) and match the original image size
        decoder_layers.append(nn.Conv2d(in_channels, 3, kernel_size=3, padding=1))  # Output to 3 channels (RGB)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        latent = self.encode(x)
        decoded = self.decode(latent)
        return latent, decoded

    def encode(self, x):
        encoded = self.encoder(x)
        latent = self.latent_conv(encoded)  # Ensure correct transformation to latent channels
        return latent

    def decode(self, latent):
        decoded = self.decoder(latent)
        return decoded

class FluxGenerator():

    @torch.inference_mode()
    def __init__(self, model=None, clip=None, vae=None):

        # self.clip, self.vae, self.model = load_flux()

        # clip_path_1 = '/home/mix/Playground/ComfyUI/models/clip/clip_l.safetensors'
        # clip_path_2 = '/home/mix/Playground/ComfyUI/models/clip/t5xxl_fp8_e4m3fn.safetensors'
        # vae_path = '/home/mix/Playground/ComfyUI/models/vae/flux-ae.safetensors'
        # unet_path = '/home/mix/Playground/ComfyUI/models/unet/flux1-schnell.safetensors'
        # self.clip, self.vae, self.model = model_manager.load_model(('/home/mix/Playground/ComfyUI/models/clip/clip_l.safetensors', '/home/mix/Playground/ComfyUI/models/clip/t5xxl_fp8_e4m3fn.safetensors'), '/home/mix/Playground/ComfyUI/models/vae/flux-ae.safetensors', '/home/mix/Playground/ComfyUI/models/unet/flux1-schnell.safetensors')
        # self.vae.device = torch.device('cuda')
        self.model, self.clip, self.vae = model, clip, vae

        # self.model.model.diffusion_model = apply_caching_and_compilation(self.model.model.diffusion_model)
        self.embedding = None
        self.noise = None
        self.seed = None
        self.guider = Guider_Basic(self.model)
        self.prompt = ""
        self.offload = False

        # self.set_offload(offload)

        self.sampler = None
        self.sampler_name = ""
        self.width = 0
        self.height = 0
        self.latent = None
        self.first_gen = True
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model = DynamicAutoencoder(latent_scale=8, latent_channels=16, in_blocks=[32, 64, 128, 256], out_blocks=[256,128,64,32]).to(device)
        # #model.encoder.load_state_dict(torch.load('/home/mix/Documents/tiny_encoder_epoch_3500.pth'))
        # model.decoder.load_state_dict(torch.load('/home/mix/Playground/flux_base/.experiments/tests/app_out/tiny_decoder_flux_290.pth'))
        # jit = True
        # # self.tiny_vae = model.half()
        # if jit:
        #     model_scripted = torch.jit.script(self.tiny_vae.decoder)
        #     #model_scripted = model_scripted.half().eval()
        #     self.tiny_vae.decoder = model_scripted
        # else:
        #     # Optional: Apply torch.compile to the model
        #     model_optimized = torch.compile(model.decoder)
        #     #model_optimized = model_optimized.eval()
        #
        #
        #     self.tiny_vae.decoder = model_optimized

    def set_offload(self, offload: bool = False):
        if self.offload != offload:
            self.offload = offload
            if self.offload:
                print("Enabling offload mode.")
                # Move models to CPU if offload is True
                self.vae.first_stage_model.to('cpu')
                self.model.model.to('cpu')
                self.clip.cond_stage_model.to('cpu')
                torch.cuda.empty_cache()
            else:
                print("Disabling offload mode.")
                # Move models to GPU if offload is False
                self.vae.first_stage_model.to('cuda')
                self.model.model.to('cuda')
                torch.cuda.empty_cache()
    @torch.inference_mode()
    def prepare_sampling(self,
                         prompt: str = "",
                         width: int = 768,
                         height: int = 768,
                         sampler_name: str = 'euler',
                         seed: int = -1
                         ):
        if seed == -1:
            seed = secrets.randbelow(18446744073709551615)
        if self.embedding == None or self.prompt != prompt:
            # if self.offload:
            #     print("LOADING CLIP TO GPU")
            # if hasattr(self.clip.cond_stage_model, 'device'):
            #     if self.clip.cond_stage_model.device != torch.device('cuda'):
            #         self.clip.cond_stage_model.to('cuda')
            #         self.clip.cond_stage_model.device = torch.device('cuda')
            self.embedding = encode(self.clip, prompt, prompt, 3.5)
            self.guider.set_conds(self.embedding)
            self.prompt = prompt
            # if self.offload:
            # if self.first_gen:
            #     self.clip.cond_stage_model.to('cpu')
            #     torch.cuda.empty_cache()
            #     self.first_gen = False
        if self.noise == None or self.seed != seed:
            self.seed = seed
            self.noise = Noise_RandomNoise(seed)
        if self.sampler == None or self.sampler_name != sampler_name:
            self.sampler = samplers.sampler_object(sampler_name)
        if self.latent == None or self.width != width or self.height != height:
            self.width = width
            self.height = height
            self.latent = {"samples": torch.zeros([1, 4, height // 8, width // 8], device=torch.device('cuda'))}

    def __call__(self,
                 prompt:str = "",
                 width:int = 768,
                 height:int = 768,
                 sampler_name:str = 'euler',
                 scheduler_name:str = 'simple',
                 steps:int = 4,
                 denoise:float=1.0,
                 output_type='pil',
                 seed:int=-1,
                 save:bool=True,
                 name:str=""):
        infer_start = time.time()

        with torch.inference_mode():
            self.prepare_sampling(prompt,
                                  width,
                                  height,
                                  sampler_name,
                                  seed)
            # if self.offload == True:
            #     print("WILL OFFLOAD VAE IN FAVOR FOR DIFFUSION")
            #     self.vae.first_stage_model.to('cpu')
            #     torch.cuda.empty_cache()
            #     self.model.model.to('cuda')
            sigmas = get_sigmas(self.model, scheduler_name, steps, denoise)
            out = sample_fn(self.noise, self.model, self.guider, self.sampler, sigmas, self.latent)

        test_tiny = False
        if output_type != 'latent':
            if not test_tiny:
                # Measure the time taken by the original VAE
                start_time = time.time()
                dtype = self.vae.get_dtype()
                with torch.no_grad():
                    out_samples = out["samples"].to(self.vae.device, dtype=dtype)
                    res = self.vae.first_stage_model.decode(out_samples)
                # Normalize and clamp the output on the GPU
                res = torch.clamp((res + 1.0) / 2.0, min=0.0, max=1.0)
                # Convert to uint8 on the GPU (assuming the range is [0, 1] for normalization)
                # Scale to [0, 255] and convert to uint8
                res = (res * 255).to(torch.uint8)
                # Step 1: Remove the batch dimension and move the color channel to the last dimension
                img_gpu = res.squeeze(0).permute(1, 2, 0)  # Now shape is (1024, 1024, 3)
                # Step 2: Move the tensor to the CPU and convert to NumPy array
                img = img_gpu.cpu().numpy()
                end_time = time.time()
                vae_time = end_time - start_time
                infer_end_time = end_time - infer_start

                print(f"Original VAE decoding time: {vae_time:.4f} seconds")
                print(f"Total Inference: {infer_end_time:.4f} seconds")

            else:
                # Measure the time taken by the TinyVAE
                start_time = time.time()
                decoded_sample = self.tiny_vae.decoder(out["samples"].to('cuda', dtype=torch.float16))
                # Postprocess and save the output image
                img = np.array(postprocess(decoded_sample[0])).astype(np.uint8)
                end_time = time.time()
                tiny_vae_time = end_time - start_time
                infer_end_time = end_time - infer_start
                print(f"TinyVAE decoding time: {tiny_vae_time:.4f} seconds")
                print(f"Total Inference: {infer_end_time:.4f} seconds")
            # img = torch.clamp(out[0] * 255, 0, 255).byte().cpu().numpy()[0]
            if save:
                if not name:
                    name = "output.png"
                else:
                    if not name.endswith(".png"):
                        name += ".png"
                import cv2
                cv2.imwrite(name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        else:
            img = out
        #self.model.model.diffusion_model.export_layers_info()
        return img

# print(NODE_REGISTRY)
#
# generator = None
# for i in range(100):
#     start_time = time.time()  # Start the timer
#     if generator == None:
#         generator = FluxGenerator(offload=False)
#
#     image = generator("Cyberpunk", steps=4, width=1024, height=1024, name=str(i))
#     end_time = time.time()  # End the timer
#     duration = end_time - start_time  # Calculate the duration
#     print(f"Image {i} took {duration:.2f} seconds to generate.")
