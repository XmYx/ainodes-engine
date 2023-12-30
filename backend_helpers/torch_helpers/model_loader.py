"""
aiNodes node engine

stable diffusion pytorch model loader

www.github.com/XmYx/ainodes-engine
miklos.mnagy@gmail.com
"""
import hashlib
import math
import os

import numpy as np
import yaml
from omegaconf import OmegaConf
from torch.nn.functional import silu

# from ldm_ainodes.models.autoencoder import AutoencoderKL
# from ldm_ainodes.modules.sub_quadratic_attention import OOM_EXCEPTION
# from . import diffusers_convert, model_base
# from .chainner_models import model_loading
#from .lora_loader import ModelPatcher
# from .sd_optimizations.sd_hijack import apply_optimizations
from .torch_gc import torch_gc
# from ldm_ainodes.util import instantiate_from_config, get_free_memory
from ainodes_frontend import singleton as gs

# from .ESRGAN import model as upscaler

import torch
from torch import nn
import safetensors.torch
# import ldm_ainodes.modules.diffusionmodules.model

# class UpscalerLoader(torch.nn.Module):
#
#     """
#     Torch Upscale model loader
#     """
#
#     def __init__(self, parent=None):
#         super().__init__()
#         self.device = gs.device
#         self.loaded_model = None
#
#     def load_model(self, file="", name=""):
#         from comfy.utils import load_torch_file
#
#         load = None
#         if self.loaded_model:
#             if self.loaded_model != name:
#                 gs.models[self.loaded_model] = None
#                 del gs.models[self.loaded_model]
#                 torch_gc()
#                 load = True
#             else:
#                 load = None
#         else:
#             load = True
#
#         if load:
#             state_dict = load_torch_file(file)
#             gs.models[name] = model_loading.load_state_dict(state_dict).eval().to(gs.device)
#             self.loaded_model = name
#
#         return self.loaded_model
def encode_tiled_(self, in_samples, tile_x=512, tile_y=512, overlap=64):
    from comfy import utils
    pixel_samples = in_samples.clone()
    steps = pixel_samples.shape[0] * utils.get_tiled_scale_steps(pixel_samples.shape[3], pixel_samples.shape[2],
                                                                 tile_x, tile_y, overlap)
    steps += pixel_samples.shape[0] * utils.get_tiled_scale_steps(pixel_samples.shape[3], pixel_samples.shape[2],
                                                                  tile_x // 2, tile_y * 2, overlap)
    steps += pixel_samples.shape[0] * utils.get_tiled_scale_steps(pixel_samples.shape[3], pixel_samples.shape[2],
                                                                  tile_x * 2, tile_y // 2, overlap)
    # pbar = utils.ProgressBar(steps)

    encode_fn = lambda a: self.first_stage_model.encode(
        2. * a.to(self.vae_dtype).to(self.device) - 1.).sample().float()
    samples = utils.tiled_scale(pixel_samples, encode_fn, tile_x, tile_y, overlap, upscale_amount=(1 / 8),
                                out_channels=4, pbar=None).clone()
    samples = samples + utils.tiled_scale(pixel_samples, encode_fn, tile_x * 2, tile_y // 2, overlap,
                                          upscale_amount=(1 / 8), out_channels=4, pbar=None).clone()
    samples = samples + utils.tiled_scale(pixel_samples, encode_fn, tile_x // 2, tile_y * 2, overlap,
                                          upscale_amount=(1 / 8), out_channels=4, pbar=None).clone()
    samples = samples / 3.0
    return samples
class ModelLoader(torch.nn.Module):
    """
    Torch SD Model Loader class
    Storage is a Singleton object
    """
    def __init__(self, parent=None):
        super().__init__()
        self.device = gs.device

    def load_checkpoint_guess_config(self, ckpt_path, output_vae=True, output_clip=True, output_clipvision=False,
                                     embedding_directory=None, style=""):
        # from comfy.model_patcher import ModelPatcher
        # from comfy.sd import load_model_weights, CLIP, VAE
        # from comfy.utils import load_torch_file, calculate_parameters
        # from comfy import model_detection, clip_vision, model_management
        # from comfy.model_management import should_use_fp16
        # from comfy.sd import VAE, load_model_weights, CLIP
        # from comfy.utils import load_torch_file
        import comfy
        ckpt_path = os.path.join(gs.prefs.checkpoints, ckpt_path)
        model, clip, vae, clipvision = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory="models/embeddings")


        return model, clip, vae, clipvision

    def load_model(self, file=None, config_name=None, inpaint=False, verbose=False, style="sdp"):
        from comfy.sd import ModelPatcher, load_model_weights, CLIP, VAE
        from comfy.utils import load_torch_file

        state_dict = None
        ckpt_path = os.path.join(gs.prefs.checkpoints, file)
        config_path = os.path.join('models/configs', config_name)

        with open(config_path, 'r') as stream:
            config = yaml.safe_load(stream)
        model_config_params = config['model']['params']
        clip_config = model_config_params['cond_stage_config']
        scale_factor = model_config_params['scale_factor']
        vae_config = model_config_params['first_stage_config']

        fp16 = False
        if "unet_config" in model_config_params:
            if "params" in model_config_params["unet_config"]:
                unet_config = model_config_params["unet_config"]["params"]
                if "use_fp16" in unet_config:
                    fp16 = unet_config["use_fp16"]

        noise_aug_config = None
        if "noise_aug_config" in model_config_params:
            noise_aug_config = model_config_params["noise_aug_config"]

        v_prediction = False

        if "parameterization" in model_config_params:
            if model_config_params["parameterization"] == "v":
                v_prediction = True
        clip = None
        vae = None

        class WeightsLoader(torch.nn.Module):
            pass

        w = WeightsLoader()
        load_state_dict_to = []
        vae = VAE(scale_factor=scale_factor, config=vae_config)
        w.first_stage_model = vae.first_stage_model
        load_state_dict_to = [w]
        #vae.first_stage_model = w.first_stage_model.cuda()

        clip = CLIP(config=clip_config, embedding_directory="models/embeddings")
        w.cond_stage_model = clip.cond_stage_model
        load_state_dict_to = [w]
        clip.cond_stage_model = w.cond_stage_model

        # model = instantiate_from_config(config.model)
        # sd = load_torch_file(ckpt_path)
        # model = load_model_weights(model, sd, verbose=False, load_state_dict_to=load_state_dict_to)
        # model = model.half()
        from comfy import model_base
        if config['model']["target"].endswith("LatentInpaintDiffusion"):
            model = model_base.SDInpaint(unet_config, v_prediction=v_prediction)
        elif config['model']["target"].endswith("ImageEmbeddingConditionedLatentDiffusion"):
            model = model_base.SD21UNCLIP(unet_config, noise_aug_config["params"], v_prediction=v_prediction)
        else:
            model = model_base.BaseModel(unet_config, v_prediction=v_prediction)

        if state_dict is None:
            state_dict = load_torch_file(ckpt_path)
        model = load_model_weights(model, state_dict, verbose=False, load_state_dict_to=load_state_dict_to)
        # enabling benchmark option seems to enable a range of cards to do fp16 when they otherwise can't
        # see https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/4407

        # if torch.cuda.is_available():
        #     if any([torch.cuda.get_device_capability(devid) == (7, 5) for devid in range(0, torch.cuda.device_count())]):
        #         torch.backends.cudnn.benchmark = True
        #
        #     torch.backends.cuda.matmul.allow_tf32 = True
        #     torch.backends.cudnn.allow_tf32 = True
        # apply_optimizations(style)
        # if style != "None":
        #     apply_optimizations(style)

        return ModelPatcher(model), clip, vae
        # gs.models["sd"] = ModelPatcher(model)
        # gs.models["clip"] = clip
        # gs.models["vae"] = vae
        #print("LOADED")
        # if gs.debug:
        #     print(gs.models["sd"],gs.models["clip"],gs.models["vae"])

    def load_model_old(self, file=None, config=None, inpaint=False, verbose=False):

        if file not in gs.loaded_models["loaded"]:
            gs.loaded_models["loaded"].append(file)
            ckpt = f"models/checkpoints/{file}"
            gs.force_inpaint = False
            ckpt_print = ckpt.replace('\\', '/')
            #config, version = self.return_model_version(ckpt)
            #if 'Inpaint' in version:
            #    gs.force_inpaint = True
            #    print("Forcing Inpaint")

            config = os.path.join('models/configs', config)
            self.prev_seamless = False
            if verbose:
                print(f"Loading model from {ckpt} with config {config}")
            config = OmegaConf.load(config)

            # print(config.model['params'])

            if 'num_heads' in config.model['params']['unet_config']['params']:
                gs.model_version = '1.5'
            elif 'num_head_channels' in config.model['params']['unet_config']['params']:
                gs.model_version = '2.0'
            if config.model['params']['conditioning_key'] == 'hybrid-adm':
                gs.model_version = '2.0'
            if 'parameterization' in config.model['params']:
                gs.model_resolution = 768
            else:
                gs.model_resolution = 512
            print(f'v {gs.model_version} found with resolution {gs.model_resolution}')
            if verbose:
                print('gs.model_version', gs.model_version)
            checkpoint_file = ckpt
            _, extension = os.path.splitext(checkpoint_file)
            map_location = "cpu"
            if extension.lower() == ".safetensors":
                pl_sd = safetensors.torch.load_file(checkpoint_file, device=map_location)
            else:
                pl_sd = torch.load(checkpoint_file, map_location=map_location)
            if "global_step" in pl_sd:
                print(f"Global Step: {pl_sd['global_step']}")
            sd = self.get_state_dict_from_checkpoint(pl_sd)


            from comfy.ldm.util import instantiate_from_config

            model = instantiate_from_config(config.model)
            m, u = model.load_state_dict(sd, strict=False)

            k = list(sd.keys())
            for x in k:
                # print(x)
                if x.startswith("cond_stage_model.transformer.") and not x.startswith(
                        "cond_stage_model.transformer.text_model."):
                    y = x.replace("cond_stage_model.transformer.", "cond_stage_model.transformer.text_model.")
                    sd[y] = sd.pop(x)

            if 'cond_stage_model.transformer.text_model.embeddings.position_ids' in sd:
                ids = sd['cond_stage_model.transformer.text_model.embeddings.position_ids']
                if ids.dtype == torch.float32:
                    sd['cond_stage_model.transformer.text_model.embeddings.position_ids'] = ids.round()

            keys_to_replace = {
                "cond_stage_model.model.positional_embedding": "cond_stage_model.transformer.text_model.embeddings.position_embedding.weight",
                "cond_stage_model.model.token_embedding.weight": "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight",
                "cond_stage_model.model.ln_final.weight": "cond_stage_model.transformer.text_model.final_layer_norm.weight",
                "cond_stage_model.model.ln_final.bias": "cond_stage_model.transformer.text_model.final_layer_norm.bias",
            }

            for x in keys_to_replace:
                if x in sd:
                    sd[keys_to_replace[x]] = sd.pop(x)

            resblock_to_replace = {
                "ln_1": "layer_norm1",
                "ln_2": "layer_norm2",
                "mlp.c_fc": "mlp.fc1",
                "mlp.c_proj": "mlp.fc2",
                "attn.out_proj": "self_attn.out_proj",
            }

            for resblock in range(24):
                for x in resblock_to_replace:
                    for y in ["weight", "bias"]:
                        k = "cond_stage_model.model.transformer.resblocks.{}.{}.{}".format(resblock, x, y)
                        k_to = "cond_stage_model.transformer.text_model.encoder.layers.{}.{}.{}".format(resblock,
                                                                                                        resblock_to_replace[
                                                                                                            x], y)
                        if k in sd:
                            sd[k_to] = sd.pop(k)

                for y in ["weight", "bias"]:
                    k_from = "cond_stage_model.model.transformer.resblocks.{}.attn.in_proj_{}".format(resblock, y)
                    if k_from in sd:
                        weights = sd.pop(k_from)
                        for x in range(3):
                            p = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
                            k_to = "cond_stage_model.transformer.text_model.encoder.layers.{}.{}.{}".format(resblock,
                                                                                                            p[x], y)
                            sd[k_to] = weights[1024 * x:1024 * (x + 1)]

            for x in []:
                x.load_state_dict(sd, strict=False)

            if len(m) > 0 and verbose:
                print("missing keys:")
                print(m)
            if len(u) > 0 and verbose:
                print("unexpected keys:")
                print(u)
            model.half()
            from comfy.model_patcher import ModelPatcher
            model = ModelPatcher(model)

            value = "sd" if inpaint == False else "inpaint"

            gs.models[value] = model
            #gs.models["sd"].cond_stage_model.device = self.device
            for m in gs.models[value].model.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    m._orig_padding_mode = m.padding_mode

            autoencoder_version = self.get_autoencoder_version()

            gs.models[value].linear_decode = make_linear_decode(autoencoder_version, self.device)
            del pl_sd
            del sd
            del m, u
            del model
            torch_gc()

            #if gs.model_version == '1.5' and not 'Inpaint' in version:
            #    self.run_post_load_model_generation_specifics()

            gs.models[value].model.eval()

            # todo make this 'cuda' a parameter
            gs.models[value].model.to(self.device)

        return ckpt
    def return_model_version(self, model):
        print('calculating sha to estimate the model version')
        with open(model, 'rb') as file:
            # Read the contents of the file
            file_contents = file.read()
            # Calculate the SHA-256 hash
            sha256_hash = hashlib.sha256(file_contents).hexdigest()
            if sha256_hash == 'd635794c1fedfdfa261e065370bea59c651fc9bfa65dc6d67ad29e11869a1824':
                version = '2.0 512'
                config = 'v2-inference.yaml'
            elif sha256_hash == '2a208a7ded5d42dcb0c0ec908b23c631002091e06afe7e76d16cd11079f8d4e3':
                version = '2.0 Inpaint'
                config = 'v2-inpainting-inference.yaml'
            elif sha256_hash == 'bfcaf0755797b0c30eb00a3787e8b423eb1f5decd8de76c4d824ac2dd27e139f':
                version = '2.0 768'
                config = 'v2-inference.yaml'
            elif sha256_hash == 'fe4efff1e174c627256e44ec2991ba279b3816e364b49f9be2abc0b3ff3f8556':
                version = '1.4'
                config = 'v1-inference_fp16.yaml'
            elif sha256_hash == 'c6bbc15e3224e6973459ba78de4998b80b50112b0ae5b5c67113d56b4e366b19':
                version = '1.5 Inpaint'
                config = 'v1-inpainting-inference.yaml'
            elif sha256_hash == 'cc6cb27103417325ff94f52b7a5d2dde45a7515b25c255d8e396c90014281516':
                version = '1.5 EMA Only'
                config = 'v1-inference_fp16.yaml'
            elif sha256_hash == '88ecb782561455673c4b78d05093494b9c539fc6bfc08f3a9a4a0dd7b0b10f36':
                version = '2.1 512'
                config = 'v2-inference.yaml'
            elif sha256_hash == 'ad2a33c361c1f593c4a1fb32ea81afce2b5bb7d1983c6b94793a26a3b54b08a0':
                version = '2.1 768'
                config = 'v2-inference-v.yaml'
            else:
                version = 'unknown'
                config = 'v1-inference_fp16.yaml'
        del file
        del file_contents
        return config, version
    def get_state_dict_from_checkpoint(self, pl_sd):
        pl_sd = pl_sd.pop("state_dict", pl_sd)
        pl_sd.pop("state_dict", None)

        sd = {}
        for k, v in pl_sd.items():
            new_key = self.transform_checkpoint_dict_key(k)

            if new_key is not None:
                sd[new_key] = v

        pl_sd.clear()
        pl_sd.update(sd)
        sd = None
        return pl_sd
    def get_autoencoder_version(self):
        return "sd-v1"  # TODO this will be different for different models

    def transform_checkpoint_dict_key(self, k):
        chckpoint_dict_replacements = {
            'cond_stage_model.transformer.embeddings.': 'cond_stage_model.transformer.text_model.embeddings.',
            'cond_stage_model.transformer.encoder.': 'cond_stage_model.transformer.text_model.encoder.',
            'cond_stage_model.transformer.final_layer_norm.': 'cond_stage_model.transformer.text_model.final_layer_norm.',
        }
        for text, replacement in chckpoint_dict_replacements.items():
            if k.startswith(text):
                k = replacement + k[len(text):]

        return k
    def load_inpaint_model(self, modelname):

        """if inpaint in model name: force inpaint
        else
        try load normal
        except error
            load inpaint"""



        """if "sd" in gs.models:
            gs.models["sd"].to('cpu')
            del gs.models["sd"]
            torch_gc()
        if "custom_model_name" in gs.models:
            del gs.models["custom_model_name"]
            torch_gc()"""
        """Load and initialize the model from configuration variables passed at object creation time"""
        if "inpaint" not in gs.models:
            weights = modelname
            config = 'models/configs/v1-inpainting-inference.yaml'
            embedding_path = None

            config = OmegaConf.load(config)
            from comfy.ldm.util import instantiate_from_config
            model = instantiate_from_config(config.model)

            model.load_state_dict(torch.load(weights)["state_dict"], strict=False)

            device = self.device
            gs.models["inpaint"] = model.half().to(device)
            del model
            return

    def load_vae(self, file):
        from comfy.sd import VAE
        path = os.path.join('models/vae', file)
        print("Loading", path)
        #gs.models["sd"].first_stage_model.cpu()
        # gs.models["sd"].first_stage_model = None
        # gs.models["sd"].first_stage_model = VAE(ckpt_path=path)
        vae = VAE(ckpt_path=path)
        print("VAE Loaded", file)
        return vae



# Decodes the image without passing through the upscaler. The resulting image will be the same size as the latent
# Thanks to Kevin Turner (https://github.com/keturn) we have a shortcut to look at the decoded image!
def make_linear_decode(model_version, device='cuda:0'):
    v1_4_rgb_latent_factors = [
        #   R       G       B
        [ 0.298,  0.207,  0.208],  # L1
        [ 0.187,  0.286,  0.173],  # L2
        [-0.158,  0.189,  0.264],  # L3
        [-0.184, -0.271, -0.473],  # L4
    ]

    if model_version[:5] == "sd-v1":
        rgb_latent_factors = torch.Tensor(v1_4_rgb_latent_factors).to(device)
    else:
        raise Exception(f"Model name {model_version} not recognized.")

    def linear_decode(latent):
        latent_image = latent.permute(0, 2, 3, 1) @ rgb_latent_factors
        latent_image = latent_image.permute(0, 3, 1, 2)
        return latent_image

    return linear_decode


# class VAE:
#     def __init__(self, ckpt_path=None, scale_factor=0.18215, device=None, config=None):
#         if config is None:
#             #default SD1.x/SD2.x VAE parameters
#             ddconfig = {'double_z': True, 'z_channels': 4, 'resolution': 256, 'in_channels': 3, 'out_ch': 3, 'ch': 128, 'ch_mult': [1, 2, 4, 4], 'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}
#             self.first_stage_model = AutoencoderKL(ddconfig, {'target': 'torch.nn.Identity'}, 4, monitor="val/rec_loss")
#         else:
#             self.first_stage_model = AutoencoderKL(**(config['params']))
#         self.first_stage_model = self.first_stage_model.eval()
#         if ckpt_path is not None:
#             sd = load_torch_file(ckpt_path)
#             if 'decoder.up_blocks.0.resnets.0.norm1.weight' in sd.keys(): #diffusers format
#                 sd = diffusers_convert.convert_vae_state_dict(sd)
#             self.first_stage_model.load_state_dict(sd, strict=False)
#
#         self.scale_factor = scale_factor
#         if device is None:
#             device = gs.device
#         self.device = device
#
#     def decode_tiled_(self, samples, tile_x=64, tile_y=64, overlap = 16):
#         steps = samples.shape[0] * get_tiled_scale_steps(samples.shape[3], samples.shape[2], tile_x, tile_y, overlap)
#         steps += samples.shape[0] * get_tiled_scale_steps(samples.shape[3], samples.shape[2], tile_x // 2, tile_y * 2, overlap)
#         steps += samples.shape[0] * get_tiled_scale_steps(samples.shape[3], samples.shape[2], tile_x * 2, tile_y // 2, overlap)
#         #pbar = utils.ProgressBar(steps)
#
#         decode_fn = lambda a: (self.first_stage_model.decode(1. / self.scale_factor * a.to(self.device)) + 1.0)
#         output = torch.clamp((
#             (tiled_scale(samples, decode_fn, tile_x // 2, tile_y * 2, overlap, upscale_amount = 8, pbar = None) +
#             tiled_scale(samples, decode_fn, tile_x * 2, tile_y // 2, overlap, upscale_amount = 8, pbar = None) +
#              tiled_scale(samples, decode_fn, tile_x, tile_y, overlap, upscale_amount = 8, pbar = None))
#             / 3.0) / 2.0, min=0.0, max=1.0)
#         return output
#
#     def decode(self, samples_in):
#
#         print(samples_in.shape)
#
#         # self.first_stage_model = self.first_stage_model.to(self.device)
#         #
#         # output = self.decode_tiled_(samples_in, 64, 64, 16)
#         # self.first_stage_model = self.first_stage_model.cpu()
#         #
#         # return output.movedim(1,-1)
#
#         #model_management.unload_model()
#         self.first_stage_model = self.first_stage_model.to(self.device)
#         try:
#             free_memory = get_free_memory(self.device)
#             batch_number = int((free_memory * 0.7) / (2562 * samples_in.shape[2] * samples_in.shape[3] * 64))
#             batch_number = max(1, batch_number)
#
#             pixel_samples = torch.empty((samples_in.shape[0], 3, round(samples_in.shape[2] * 8), round(samples_in.shape[3] * 8)), device="cpu")
#             for x in range(0, samples_in.shape[0], batch_number):
#                 samples = samples_in[x:x+batch_number].to(self.device)
#                 pixel_samples[x:x+batch_number] = torch.clamp((self.first_stage_model.decode(1. / self.scale_factor * samples) + 1.0) / 2.0, min=0.0, max=1.0).cpu()
#         except OOM_EXCEPTION as e:
#             print("Warning: Ran out of memory when regular VAE decoding, retrying with tiled VAE decoding.")
#             pixel_samples = self.decode_tiled_(samples_in)
#
#         self.first_stage_model = self.first_stage_model.cpu()
#         pixel_samples = pixel_samples.cpu().movedim(1,-1)
#
#         return pixel_samples
#
#     def decode_tiled(self, samples, tile_x=64, tile_y=64, overlap = 16):
#         #model_management.unload_model()
#         self.first_stage_model = self.first_stage_model.to(self.device)
#         output = self.decode_tiled_(samples, tile_x, tile_y, overlap)
#         self.first_stage_model = self.first_stage_model.cpu()
#         return output.movedim(1,-1)
#
#     def encode(self, pixel_samples):
#         #model_management.unload_model()
#         self.first_stage_model = self.first_stage_model.to(self.device)
#         pixel_samples = pixel_samples.movedim(-1,1).to(self.device)
#         samples = self.first_stage_model.encode(2. * pixel_samples - 1.).sample() * self.scale_factor
#         self.first_stage_model = self.first_stage_model.cpu()
#         samples = samples.cpu()
#         return samples
#
#     def encode_tiled(self, pixel_samples, tile_x=512, tile_y=512, overlap = 64):
#         #model_management.unload_model()
#         self.first_stage_model = self.first_stage_model.to(self.device)
#         pixel_samples = pixel_samples.movedim(-1,1).to(self.device)
#
#         steps = pixel_samples.shape[0] * get_tiled_scale_steps(pixel_samples.shape[3], pixel_samples.shape[2], tile_x, tile_y, overlap)
#         steps += pixel_samples.shape[0] * get_tiled_scale_steps(pixel_samples.shape[3], pixel_samples.shape[2], tile_x // 2, tile_y * 2, overlap)
#         steps += pixel_samples.shape[0] * get_tiled_scale_steps(pixel_samples.shape[3], pixel_samples.shape[2], tile_x * 2, tile_y // 2, overlap)
#
#         samples = tiled_scale(pixel_samples, lambda a: self.first_stage_model.encode(2. * a - 1.).sample() * self.scale_factor, tile_x, tile_y, overlap, upscale_amount = (1/8), out_channels=4, pbar=None)
#         samples += tiled_scale(pixel_samples, lambda a: self.first_stage_model.encode(2. * a - 1.).sample() * self.scale_factor, tile_x * 2, tile_y // 2, overlap, upscale_amount = (1/8), out_channels=4, pbar=None)
#         samples += tiled_scale(pixel_samples, lambda a: self.first_stage_model.encode(2. * a - 1.).sample() * self.scale_factor, tile_x // 2, tile_y * 2, overlap, upscale_amount = (1/8), out_channels=4, pbar=None)
#         samples /= 3.0
#         self.first_stage_model = self.first_stage_model.cpu()
#         samples = samples.cpu()
#         return samples
#
#
#
# class CLIP:
#     def __init__(self, config={}, embedding_directory=None, no_init=False):
#         if no_init:
#             return
#         self.target_clip = config["target"]
#         if "params" in config:
#             params = config["params"]
#         else:
#             params = {}
#
#         if self.target_clip == "ldm.modules.encoders.modules.FrozenOpenCLIPEmbedder":
#             clip = SD2ClipModel
#             tokenizer = SD2Tokenizer
#         elif self.target_clip == "ldm.modules.encoders.modules.FrozenCLIPEmbedder":
#             clip = SD1ClipModel
#             tokenizer = SD1Tokenizer
#
#         self.cond_stage_model = clip(**(params))
#         self.tokenizer = tokenizer(embedding_directory=embedding_directory)
#         self.patcher = ModelPatcher(self.cond_stage_model)
#         self.layer_idx = -1
#
#     def clone(self):
#         n = CLIP(no_init=True)
#         n.target_clip = self.target_clip
#         n.patcher = self.patcher.clone()
#         n.cond_stage_model = self.cond_stage_model
#         n.tokenizer = self.tokenizer
#         n.layer_idx = self.layer_idx
#         return n
#
#     def load_from_state_dict(self, sd):
#         self.cond_stage_model.transformer.load_state_dict(sd, strict=False)
#
#     def add_patches(self, patches, strength=1.0):
#         return self.patcher.add_patches(patches, strength)
#
#     def clip_layer(self, layer_idx):
#         self.layer_idx = layer_idx
#
#     def encode(self, text):
#         self.cond_stage_model.clip_layer(self.layer_idx)
#         tokens = self.tokenizer.tokenize_with_weights(text)
#         try:
#             self.patcher.patch_model()
#             cond = self.cond_stage_model.encode_token_weights(tokens)
#             self.patcher.unpatch_model()
#         except Exception as e:
#             self.patcher.unpatch_model()
#             raise e
#         return cond
#
#
# import os
#
# from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextConfig
# import torch
#
# class ClipTokenWeightEncoderSD1:
#     def encode_token_weights(self, token_weight_pairs):
#         z_empty = self.encode(self.empty_tokens)
#         output = []
#         for x in token_weight_pairs:
#             tokens = [list(map(lambda a: a[0], x))]
#             z = self.encode(tokens)
#             for i in range(len(z)):
#                 for j in range(len(z[i])):
#                     weight = x[j][1]
#                     z[i][j] = (z[i][j] - z_empty[0][j]) * weight + z_empty[0][j]
#             output += [z]
#         if (len(output) == 0):
#             return self.encode(self.empty_tokens)
#         return torch.cat(output, dim=-2)
#
# class SD1ClipModel(torch.nn.Module, ClipTokenWeightEncoderSD1):
#     """Uses the CLIP transformer encoder for text (from huggingface)"""
#     LAYERS = [
#         "last",
#         "pooled",
#         "hidden"
#     ]
#     def __init__(self, version="openai/clip-vit-large-patch14", device="cpu", max_length=77,
#                  freeze=True, layer="last", layer_idx=None, textmodel_json_config=None, textmodel_path=None):  # clip-vit-base-patch32
#         super().__init__()
#         assert layer in self.LAYERS
#         if textmodel_path is not None:
#             self.transformer = CLIPTextModel.from_pretrained(textmodel_path)
#         else:
#             if textmodel_json_config is None:
#                 textmodel_json_config = os.path.join("models/configs/sd1_clip_config.json")
#             config = CLIPTextConfig.from_json_file(textmodel_json_config)
#             self.transformer = CLIPTextModel(config)
#
#         self.device = device
#         self.max_length = max_length
#         if freeze:
#             self.freeze()
#         self.layer = layer
#         self.layer_idx = None
#         self.empty_tokens = [[49406] + [49407] * 76]
#         if layer == "hidden":
#             assert layer_idx is not None
#             assert abs(layer_idx) <= 12
#             self.clip_layer(layer_idx)
#
#     def freeze(self):
#         self.transformer = self.transformer.eval()
#         #self.train = disabled_train
#         for param in self.parameters():
#             param.requires_grad = False
#
#     def clip_layer(self, layer_idx):
#         if abs(layer_idx) >= 12:
#             self.layer = "last"
#         else:
#             self.layer = "hidden"
#             self.layer_idx = layer_idx
#
#     def set_up_textual_embeddings(self, tokens, current_embeds):
#         out_tokens = []
#         next_new_token = token_dict_size = current_embeds.weight.shape[0]
#         embedding_weights = []
#
#         for x in tokens:
#             tokens_temp = []
#             for y in x:
#                 if isinstance(y, int):
#                     tokens_temp += [y]
#                 else:
#                     embedding_weights += [y]
#                     tokens_temp += [next_new_token]
#                     next_new_token += 1
#             out_tokens += [tokens_temp]
#         if len(embedding_weights) > 0:
#             new_embedding = torch.nn.Embedding(next_new_token, current_embeds.weight.shape[1])
#             new_embedding.weight[:token_dict_size] = current_embeds.weight[:]
#             n = token_dict_size
#             for x in embedding_weights:
#                 new_embedding.weight[n] = x
#                 n += 1
#             self.transformer.set_input_embeddings(new_embedding)
#         return out_tokens
#
#     def forward(self, tokens):
#         backup_embeds = self.transformer.get_input_embeddings()
#         tokens = self.set_up_textual_embeddings(tokens, backup_embeds)
#         tokens = torch.LongTensor(tokens).to(self.device)
#         outputs = self.transformer(input_ids=tokens, output_hidden_states=self.layer=="hidden")
#         self.transformer.set_input_embeddings(backup_embeds)
#
#         if self.layer == "last":
#             z = outputs.last_hidden_state
#         elif self.layer == "pooled":
#             z = outputs.pooler_output[:, None, :]
#         else:
#             z = outputs.hidden_states[self.layer_idx]
#             z = self.transformer.text_model.final_layer_norm(z)
#
#         return z
#
#     def encode(self, tokens):
#         return self(tokens)
#
# def parse_parentheses(string):
#     result = []
#     current_item = ""
#     nesting_level = 0
#     for char in string:
#         if char == "(":
#             if nesting_level == 0:
#                 if current_item:
#                     result.append(current_item)
#                     current_item = "("
#                 else:
#                     current_item = "("
#             else:
#                 current_item += char
#             nesting_level += 1
#         elif char == ")":
#             nesting_level -= 1
#             if nesting_level == 0:
#                 result.append(current_item + ")")
#                 current_item = ""
#             else:
#                 current_item += char
#         else:
#             current_item += char
#     if current_item:
#         result.append(current_item)
#     return result
#
# def token_weights(string, current_weight):
#     a = parse_parentheses(string)
#     out = []
#     for x in a:
#         weight = current_weight
#         if len(x) >= 2 and x[-1] == ')' and x[0] == '(':
#             x = x[1:-1]
#             xx = x.rfind(":")
#             weight *= 1.1
#             if xx > 0:
#                 try:
#                     weight = float(x[xx+1:])
#                     x = x[:xx]
#                 except:
#                     pass
#             out += token_weights(x, weight)
#         else:
#             out += [(x, current_weight)]
#     return out
#
# def escape_important(text):
#     text = text.replace("\\)", "\0\1")
#     text = text.replace("\\(", "\0\2")
#     return text
#
# def unescape_important(text):
#     text = text.replace("\0\1", ")")
#     text = text.replace("\0\2", "(")
#     return text
#
# def load_embed(embedding_name, embedding_directory):
#     embed_path = os.path.join(embedding_directory, embedding_name)
#     if not os.path.isfile(embed_path):
#         extensions = ['.safetensors', '.pt', '.bin']
#         valid_file = None
#         for x in extensions:
#             t = embed_path + x
#             if os.path.isfile(t):
#                 valid_file = t
#                 break
#         if valid_file is None:
#             return None
#         else:
#             embed_path = valid_file
#
#     if embed_path.lower().endswith(".safetensors"):
#         import safetensors.torch
#         embed = safetensors.torch.load_file(embed_path, device="cpu")
#     else:
#         if 'weights_only' in torch.load.__code__.co_varnames:
#             embed = torch.load(embed_path, weights_only=True, map_location="cpu")
#         else:
#             embed = torch.load(embed_path, map_location="cpu")
#     if 'string_to_param' in embed:
#         values = embed['string_to_param'].values()
#     else:
#         values = embed.values()
#     return next(iter(values))
#
# class SD1Tokenizer:
#     def __init__(self, tokenizer_path=None, max_length=77, pad_with_end=True, embedding_directory=None):
#         if tokenizer_path is None:
#             tokenizer_path = os.path.join("models/configs/sd1_tokenizer")
#         self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
#         self.max_length = max_length
#         self.max_tokens_per_section = self.max_length - 2
#
#         empty = self.tokenizer('')["input_ids"]
#         self.start_token = empty[0]
#         self.end_token = empty[1]
#         self.pad_with_end = pad_with_end
#         vocab = self.tokenizer.get_vocab()
#         self.inv_vocab = {v: k for k, v in vocab.items()}
#         self.embedding_directory = gs.prefs.embeddings
#         self.max_word_length = 8
#
#     def tokenize_with_weights(self, text):
#         text = escape_important(text)
#         parsed_weights = token_weights(text, 1.0)
#
#         tokens = []
#         for t in parsed_weights:
#             to_tokenize = unescape_important(t[0]).replace("\n", " ").split(' ')
#             while len(to_tokenize) > 0:
#                 word = to_tokenize.pop(0)
#                 temp_tokens = []
#                 embedding_identifier = "embedding:"
#                 if word.startswith(embedding_identifier) and self.embedding_directory is not None:
#                     embedding_name = word[len(embedding_identifier):].strip('\n')
#
#                     print("EMBED NAME", embedding_name)
#
#                     embed = load_embed(embedding_name, self.embedding_directory)
#                     if embed is None:
#                         stripped = embedding_name.strip(',')
#                         if len(stripped) < len(embedding_name):
#                             embed = load_embed(stripped, self.embedding_directory)
#                             if embed is not None:
#                                 to_tokenize.insert(0, embedding_name[len(stripped):])
#
#                     if embed is not None:
#                         if len(embed.shape) == 1:
#                             temp_tokens += [(embed, t[1])]
#                         else:
#                             for x in range(embed.shape[0]):
#                                 temp_tokens += [(embed[x], t[1])]
#                     else:
#                         print("warning, embedding:{} does not exist, ignoring".format(embedding_name))
#                 elif len(word) > 0:
#                     tt = self.tokenizer(word)["input_ids"][1:-1]
#                     for x in tt:
#                         temp_tokens += [(x, t[1])]
#                 tokens_left = self.max_tokens_per_section - (len(tokens) % self.max_tokens_per_section)
#
#                 #try not to split words in different sections
#                 if tokens_left < len(temp_tokens) and len(temp_tokens) < (self.max_word_length):
#                     for x in range(tokens_left):
#                         tokens += [(self.end_token, 1.0)]
#                 tokens += temp_tokens
#
#         out_tokens = []
#         for x in range(0, len(tokens), self.max_tokens_per_section):
#             o_token = [(self.start_token, 1.0)] + tokens[x:min(self.max_tokens_per_section + x, len(tokens))]
#             o_token += [(self.end_token, 1.0)]
#             if self.pad_with_end:
#                 o_token +=[(self.end_token, 1.0)] * (self.max_length - len(o_token))
#             else:
#                 o_token +=[(0, 1.0)] * (self.max_length - len(o_token))
#
#             out_tokens += [o_token]
#
#         return out_tokens
#
#     def untokenize(self, token_weight_pair):
#         return list(map(lambda a: (a, self.inv_vocab[a[0]]), token_weight_pair))
#
#
# class SD2ClipModel(SD1ClipModel):
#     def __init__(self, arch="ViT-H-14", device="cpu", max_length=77, freeze=True, layer="penultimate", layer_idx=None):
#         textmodel_json_config = os.path.join("models/configs/sd2_clip_config.json")
#         super().__init__(device=device, freeze=freeze, textmodel_json_config=textmodel_json_config)
#         self.empty_tokens = [[49406] + [49407] + [0] * 75]
#         if layer == "last":
#             pass
#         elif layer == "penultimate":
#             layer_idx = -1
#             self.clip_layer(layer_idx)
#         elif self.layer == "hidden":
#             assert layer_idx is not None
#             assert abs(layer_idx) < 24
#             self.clip_layer(layer_idx)
#         else:
#             raise NotImplementedError()
#
#     def clip_layer(self, layer_idx):
#         if layer_idx < 0:
#             layer_idx -= 1 #The real last layer of SD2.x clip is the penultimate one. The last one might contain garbage.
#         if abs(layer_idx) >= 24:
#             self.layer = "hidden"
#             self.layer_idx = -2
#         else:
#             self.layer = "hidden"
#             self.layer_idx = layer_idx
#
# class SD2Tokenizer(SD1Tokenizer):
#     def __init__(self, tokenizer_path=None, embedding_directory=None):
#         super().__init__(tokenizer_path, pad_with_end=False, embedding_directory=embedding_directory)
#
#
# def load_torch_file(ckpt, safe_load=False):
#     if ckpt.lower().endswith(".safetensors"):
#         import safetensors.torch
#         sd = safetensors.torch.load_file(ckpt, device="cpu")
#     else:
#         if safe_load:
#             if not 'weights_only' in torch.load.__code__.co_varnames:
#                 print("Warning torch.load doesn't support weights_only on this pytorch version, loading unsafely.")
#                 safe_load = False
#         if safe_load:
#             pl_sd = torch.load(ckpt, map_location="cpu", weights_only=True)
#         else:
#             pl_sd = torch.load(ckpt, map_location="cpu")
#         if "global_step" in pl_sd:
#             print(f"Global Step: {pl_sd['global_step']}")
#         if "state_dict" in pl_sd:
#             sd = pl_sd["state_dict"]
#         else:
#             sd = pl_sd
#     return sd
#
# def load_model_weights(model, sd, verbose=False, load_state_dict_to=[]):
#     m, u = model.load_state_dict(sd, strict=False)
#
#     k = list(sd.keys())
#     for x in k:
#         # print(x)
#         if x.startswith("cond_stage_model.transformer.") and not x.startswith("cond_stage_model.transformer.text_model."):
#             y = x.replace("cond_stage_model.transformer.", "cond_stage_model.transformer.text_model.")
#             sd[y] = sd.pop(x)
#
#     if 'cond_stage_model.transformer.text_model.embeddings.position_ids' in sd:
#         ids = sd['cond_stage_model.transformer.text_model.embeddings.position_ids']
#         if ids.dtype == torch.float32:
#             sd['cond_stage_model.transformer.text_model.embeddings.position_ids'] = ids.round()
#
#     keys_to_replace = {
#         "cond_stage_model.model.positional_embedding": "cond_stage_model.transformer.text_model.embeddings.position_embedding.weight",
#         "cond_stage_model.model.token_embedding.weight": "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight",
#         "cond_stage_model.model.ln_final.weight": "cond_stage_model.transformer.text_model.final_layer_norm.weight",
#         "cond_stage_model.model.ln_final.bias": "cond_stage_model.transformer.text_model.final_layer_norm.bias",
#     }
#
#     for x in keys_to_replace:
#         if x in sd:
#             sd[keys_to_replace[x]] = sd.pop(x)
#
#     resblock_to_replace = {
#         "ln_1": "layer_norm1",
#         "ln_2": "layer_norm2",
#         "mlp.c_fc": "mlp.fc1",
#         "mlp.c_proj": "mlp.fc2",
#         "attn.out_proj": "self_attn.out_proj",
#     }
#
#     for resblock in range(24):
#         for x in resblock_to_replace:
#             for y in ["weight", "bias"]:
#                 k = "cond_stage_model.model.transformer.resblocks.{}.{}.{}".format(resblock, x, y)
#                 k_to = "cond_stage_model.transformer.text_model.encoder.layers.{}.{}.{}".format(resblock, resblock_to_replace[x], y)
#                 if k in sd:
#                     sd[k_to] = sd.pop(k)
#
#         for y in ["weight", "bias"]:
#             k_from = "cond_stage_model.model.transformer.resblocks.{}.attn.in_proj_{}".format(resblock, y)
#             if k_from in sd:
#                 weights = sd.pop(k_from)
#                 for x in range(3):
#                     p = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
#                     k_to = "cond_stage_model.transformer.text_model.encoder.layers.{}.{}.{}".format(resblock, p[x], y)
#                     sd[k_to] = weights[1024*x:1024*(x + 1)]
#
#     for x in load_state_dict_to:
#         x.load_state_dict(sd, strict=False)
#
#     if len(m) > 0 and verbose:
#         print("missing keys:")
#         print(m)
#     if len(u) > 0 and verbose:
#         print("unexpected keys:")
#         print(u)
#
#     model.eval()
#     return model
#
#
# def load_torch_file(ckpt, safe_load=False):
#     if ckpt.lower().endswith(".safetensors"):
#         import safetensors.torch
#         sd = safetensors.torch.load_file(ckpt, device="cpu")
#     else:
#         if safe_load:
#             if not 'weights_only' in torch.load.__code__.co_varnames:
#                 print("Warning torch.load doesn't support weights_only on this pytorch version, loading unsafely.")
#                 safe_load = False
#         if safe_load:
#             pl_sd = torch.load(ckpt, map_location="cpu", weights_only=True)
#         else:
#             pl_sd = torch.load(ckpt, map_location="cpu")
#         if "global_step" in pl_sd:
#             print(f"Global Step: {pl_sd['global_step']}")
#         if "state_dict" in pl_sd:
#             sd = pl_sd["state_dict"]
#         else:
#             sd = pl_sd
#     return sd
#
#
# def transformers_convert(sd, prefix_from, prefix_to, number):
#     resblock_to_replace = {
#         "ln_1": "layer_norm1",
#         "ln_2": "layer_norm2",
#         "mlp.c_fc": "mlp.fc1",
#         "mlp.c_proj": "mlp.fc2",
#         "attn.out_proj": "self_attn.out_proj",
#     }
#
#     for resblock in range(number):
#         for x in resblock_to_replace:
#             for y in ["weight", "bias"]:
#                 k = "{}.transformer.resblocks.{}.{}.{}".format(prefix_from, resblock, x, y)
#                 k_to = "{}.encoder.layers.{}.{}.{}".format(prefix_to, resblock, resblock_to_replace[x], y)
#                 if k in sd:
#                     sd[k_to] = sd.pop(k)
#
#         for y in ["weight", "bias"]:
#             k_from = "{}.transformer.resblocks.{}.attn.in_proj_{}".format(prefix_from, resblock, y)
#             if k_from in sd:
#                 weights = sd.pop(k_from)
#                 shape_from = weights.shape[0] // 3
#                 for x in range(3):
#                     p = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
#                     k_to = "{}.encoder.layers.{}.{}.{}".format(prefix_to, resblock, p[x], y)
#                     sd[k_to] = weights[shape_from * x:shape_from * (x + 1)]
#     return sd


def bislerp(samples, width, height):
    def slerp(b1, b2, r):
        '''slerps batches b1, b2 according to ratio r, batches should be flat e.g. NxC'''

        c = b1.shape[-1]

        # norms
        b1_norms = torch.norm(b1, dim=-1, keepdim=True)
        b2_norms = torch.norm(b2, dim=-1, keepdim=True)

        # normalize
        b1_normalized = b1 / b1_norms
        b2_normalized = b2 / b2_norms

        # zero when norms are zero
        b1_normalized[b1_norms.expand(-1, c) == 0.0] = 0.0
        b2_normalized[b2_norms.expand(-1, c) == 0.0] = 0.0

        # slerp
        dot = (b1_normalized * b2_normalized).sum(1)
        omega = torch.acos(dot)
        so = torch.sin(omega)

        # technically not mathematically correct, but more pleasing?
        res = (torch.sin((1.0 - r.squeeze(1)) * omega) / so).unsqueeze(1) * b1_normalized + (
                    torch.sin(r.squeeze(1) * omega) / so).unsqueeze(1) * b2_normalized
        res *= (b1_norms * (1.0 - r) + b2_norms * r).expand(-1, c)

        # edge cases for same or polar opposites
        res[dot > 1 - 1e-5] = b1[dot > 1 - 1e-5]
        res[dot < 1e-5 - 1] = (b1 * (1.0 - r) + b2 * r)[dot < 1e-5 - 1]
        return res

    def generate_bilinear_data(length_old, length_new):
        coords_1 = torch.arange(length_old).reshape((1, 1, 1, -1)).to(torch.float32)
        coords_1 = torch.nn.functional.interpolate(coords_1, size=(1, length_new), mode="bilinear")
        ratios = coords_1 - coords_1.floor()
        coords_1 = coords_1.to(torch.int64)

        coords_2 = torch.arange(length_old).reshape((1, 1, 1, -1)).to(torch.float32) + 1
        coords_2[:, :, :, -1] -= 1
        coords_2 = torch.nn.functional.interpolate(coords_2, size=(1, length_new), mode="bilinear")
        coords_2 = coords_2.to(torch.int64)
        return ratios, coords_1, coords_2

    n, c, h, w = samples.shape
    h_new, w_new = (height, width)

    # linear w
    ratios, coords_1, coords_2 = generate_bilinear_data(w, w_new)
    coords_1 = coords_1.expand((n, c, h, -1))
    coords_2 = coords_2.expand((n, c, h, -1))
    ratios = ratios.expand((n, 1, h, -1))

    pass_1 = samples.gather(-1, coords_1).movedim(1, -1).reshape((-1, c))
    pass_2 = samples.gather(-1, coords_2).movedim(1, -1).reshape((-1, c))
    ratios = ratios.movedim(1, -1).reshape((-1, 1))

    result = slerp(pass_1, pass_2, ratios)
    result = result.reshape(n, h, w_new, c).movedim(-1, 1)

    # linear h
    ratios, coords_1, coords_2 = generate_bilinear_data(h, h_new)
    coords_1 = coords_1.reshape((1, 1, -1, 1)).expand((n, c, -1, w_new))
    coords_2 = coords_2.reshape((1, 1, -1, 1)).expand((n, c, -1, w_new))
    ratios = ratios.reshape((1, 1, -1, 1)).expand((n, 1, -1, w_new))

    pass_1 = result.gather(-2, coords_1).movedim(1, -1).reshape((-1, c))
    pass_2 = result.gather(-2, coords_2).movedim(1, -1).reshape((-1, c))
    ratios = ratios.movedim(1, -1).reshape((-1, 1))

    result = slerp(pass_1, pass_2, ratios)
    result = result.reshape(n, h_new, w_new, c).movedim(-1, 1)
    return result


def common_upscale(samples, width, height, upscale_method, crop):
    if crop == "center":
        old_width = samples.shape[3]
        old_height = samples.shape[2]
        old_aspect = old_width / old_height
        new_aspect = width / height
        x = 0
        y = 0
        if old_aspect > new_aspect:
            x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
        elif old_aspect < new_aspect:
            y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
        s = samples[:, :, y:old_height - y, x:old_width - x]
    else:
        s = samples

    if upscale_method == "bislerp":
        return bislerp(s, width, height)
    else:
        return torch.nn.functional.interpolate(s, size=(height, width), mode=upscale_method)


def get_tiled_scale_steps(width, height, tile_x, tile_y, overlap):
    return math.ceil((height / (tile_y - overlap))) * math.ceil((width / (tile_x - overlap)))


@torch.inference_mode()
def tiled_scale(samples, function, tile_x=64, tile_y=64, overlap=8, upscale_amount=4, out_channels=3, pbar=None):
    output = torch.empty((samples.shape[0], out_channels, round(samples.shape[2] * upscale_amount),
                          round(samples.shape[3] * upscale_amount)), device="cpu")
    for b in range(samples.shape[0]):
        s = samples[b:b + 1]
        out = torch.zeros(
            (s.shape[0], out_channels, round(s.shape[2] * upscale_amount), round(s.shape[3] * upscale_amount)),
            device="cpu")
        out_div = torch.zeros(
            (s.shape[0], out_channels, round(s.shape[2] * upscale_amount), round(s.shape[3] * upscale_amount)),
            device="cpu")
        for y in range(0, s.shape[2], tile_y - overlap):
            for x in range(0, s.shape[3], tile_x - overlap):
                s_in = s[:, :, y:y + tile_y, x:x + tile_x]

                ps = function(s_in).cpu()
                mask = torch.ones_like(ps)
                feather = round(overlap * upscale_amount)
                for t in range(feather):
                    mask[:, :, t:1 + t, :] *= ((1.0 / feather) * (t + 1))
                    mask[:, :, mask.shape[2] - 1 - t: mask.shape[2] - t, :] *= ((1.0 / feather) * (t + 1))
                    mask[:, :, :, t:1 + t] *= ((1.0 / feather) * (t + 1))
                    mask[:, :, :, mask.shape[3] - 1 - t: mask.shape[3] - t] *= ((1.0 / feather) * (t + 1))
                out[:, :, round(y * upscale_amount):round((y + tile_y) * upscale_amount),
                round(x * upscale_amount):round((x + tile_x) * upscale_amount)] += ps * mask
                out_div[:, :, round(y * upscale_amount):round((y + tile_y) * upscale_amount),
                round(x * upscale_amount):round((x + tile_x) * upscale_amount)] += mask
                if pbar is not None:
                    pbar.update(1)

        output[b:b + 1] = out / out_div
    return output
