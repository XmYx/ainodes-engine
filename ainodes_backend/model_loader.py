"""
aiNodes node engine

stable diffusion pytorch model loader

www.github.com/XmYx/ainodes-engine
miklos.mnagy@gmail.com
"""
import hashlib
import os

from omegaconf import OmegaConf

from ainodes_backend.lora_loader import ModelPatcher
from ainodes_backend.torch_gc import torch_gc
from ldm.util import instantiate_from_config
from ainodes_backend import singleton as gs


import torch
from torch import nn
import safetensors.torch


class ModelLoader(torch.nn.Module):
    """
    Torch SD Model Loader class
    Storage is a Singleton object
    """
    def __init__(self, parent=None):
        super().__init__()
        self.device = "cuda"
        print("PyTorch model loader")


    def load_model(self, file=None, config=None, inpaint=False, verbose=False):


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

            model = instantiate_from_config(config.model)

            model.load_state_dict(torch.load(weights)["state_dict"], strict=False)

            device = self.device
            gs.models["inpaint"] = model.half().to(device)
            del model
            return

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
