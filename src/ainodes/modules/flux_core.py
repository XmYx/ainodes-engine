import os
import sys
import torch



from modules import utils
from modules.sampling import sample, samplers
from modules.ldm.unet import sd
# from modules.model_loading import model_management
from modules.sampling.samplers import CFGGuider
import safetensors.torch

from modules.taesd.taesd import TAESD


class Guider_Basic(CFGGuider):
    def set_conds(self, positive):
        self.inner_set_conds({"positive": positive})


def conditioning_set_values(conditioning, values={}):
    c = []
    for t in conditioning:
        n = [t[0], t[1].copy()]
        for k in values:
            n[1][k] = values[k]
        c.append(n)

    return c

def encode(clip, clip_l, t5xxl, guidance):
    tokens = clip.tokenize(clip_l)
    tokens["t5xxl"] = clip.tokenize(t5xxl)["t5xxl"]
    output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
    cond = output.pop("cond")
    output["guidance"] = guidance
    return [[cond, output]]

def sample_fn(noise, model, guider, sampler, sigmas, latent_image):
    latent = latent_image
    latent_image = latent["samples"]
    latent = latent.copy()
    latent_image = sample.fix_empty_latent_channels(model, latent_image)
    latent["samples"] = latent_image

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    x0_output = {}
    #callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1, x0_output)

    disable_pbar = not utils.PROGRESS_BAR_ENABLED
    samples = guider.sample(model, noise.generate_noise(latent), latent_image, sampler, sigmas, denoise_mask=noise_mask, callback=None, disable_pbar=disable_pbar, seed=noise.seed)
    #samples = samples.to(model_management.intermediate_device())

    out = latent.copy()
    out["samples"] = samples
    # if "x0" in x0_output:
    #     out_denoised = latent.copy()
    #     out_denoised["samples"] = model.model.process_latent_out(x0_output["x0"].cpu())
    # else:
    #     out_denoised = out

    return out

def get_sigmas(model, scheduler, steps, denoise):
    total_steps = steps
    if denoise < 1.0:
        if denoise <= 0.0:
            return (torch.FloatTensor([]),)
        total_steps = int(steps/denoise)

    sigmas = samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, total_steps)
    sigmas = sigmas[-(steps + 1):]
    return sigmas
class Noise_RandomNoise:
    def __init__(self, seed):
        self.seed = seed

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]
        batch_inds = input_latent["batch_index"] if "batch_index" in input_latent else None
        return sample.prepare_noise(latent_image, self.seed, batch_inds)

import threading


def save_model_diff(dev_model, schnell_model, output_path):
    dev_state_dict = dev_model.state_dict()
    schnell_state_dict = schnell_model.state_dict()

    diff_state_dict = {}

    for key in dev_state_dict:
        if key in schnell_state_dict:
            if dev_state_dict[key].shape != schnell_state_dict[key].shape:
                # If shapes differ, save both tensors under different keys
                diff_state_dict[key + '_dev'] = dev_state_dict[key]
                diff_state_dict[key + '_schnell'] = schnell_state_dict[key]
            elif not torch.equal(dev_state_dict[key], schnell_state_dict[key]):
                # If shapes are the same but contents differ, store the difference
                diff_state_dict[key] = schnell_state_dict[key] - dev_state_dict[key]
        else:
            # If the key is not in the schnell model, store the entire tensor from the dev model
            diff_state_dict[key] = dev_state_dict[key]

    # Add any additional tensors that are in the schnell model but not in the dev model
    for key in schnell_state_dict:
        if key not in dev_state_dict:
            diff_state_dict[key] = schnell_state_dict[key]

    # Save the diff state dict to a safetensors file
    safetensors.torch.save_file(diff_state_dict, output_path)

    print(f"Model differences saved to {output_path}")


def apply_diff(model, diff_path, reverse=False):
    model_state_dict = model.state_dict()
    diff_state_dict = safetensors.torch.load_file(diff_path)

    for key in diff_state_dict:
        if key in model_state_dict:
            if reverse:
                model_state_dict[key] = model_state_dict[key] - diff_state_dict[key]
            else:
                model_state_dict[key] = model_state_dict[key] + diff_state_dict[key]
        else:
            model_state_dict[key] = diff_state_dict[key]

    model.load_state_dict(model_state_dict)


def convert_dev_to_schnell(dev_model, diff_path):
    apply_diff(dev_model, diff_path, reverse=False)


def convert_schnell_to_dev(schnell_model, diff_path):
    apply_diff(schnell_model, diff_path, reverse=True)


def save_model_diff_from_paths(dev_path, schnell_path, diff_output_path, dtype=None):
    dev_model = sd.load_unet(dev_path, dtype)
    schnell_model = sd.load_unet(schnell_path, dtype)

    save_model_diff(dev_model.model, schnell_model.model, diff_output_path)


class ModelManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelManager, cls).__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.loaded_models = {}
        self.max_vram = self._get_max_vram()
        self.used_vram = self.calculate_total_used_vram()

    def _get_max_vram(self):
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(device)
            return properties.total_memory
        else:
            raise RuntimeError("CUDA is not available. Please ensure that a CUDA-capable device is present.")

    def get_available_vram(self):
        return self.max_vram - self.used_vram

    def calculate_total_used_vram(self):
        return sum(
            self.calculate_model_size(model)
            for _, (clip, vae, model) in self.loaded_models.items()
            for model in [clip.cond_stage_model, vae.first_stage_model, model.model]
            if next(model.parameters()).is_cuda
        )

    def calculate_model_size(self, model):
        """Calculate the size of a model in bytes."""
        return sum(param.nelement() * param.element_size() for param in model.parameters())

    def offload_models(self, required_space, exclude_models=[]):
        """
        Offload models until required_space is available, excluding the specified models.

        Parameters:
        - required_space: int, amount of space needed in bytes.
        - exclude_models: list of models to exclude from offloading.
        """
        models_sorted_by_size = sorted(
            [(clip.cond_stage_model, self.calculate_model_size(clip.cond_stage_model)),
             (vae.first_stage_model, self.calculate_model_size(vae.first_stage_model)),
             (model.model, self.calculate_model_size(model.model))]
            for _, (clip, vae, model) in self.loaded_models.items()
            if not any(exclude is clip.cond_stage_model or exclude is vae.first_stage_model or exclude is model.model for exclude in exclude_models)
        )

        for model, size in models_sorted_by_size:
            if self.get_available_vram() >= required_space:
                break

            self.offload_model(model)
            self.update_used_vram()
    @torch.inference_mode()
    def offload_model(self, model):
        """Offload a specific model to free up VRAM."""
        if next(model.parameters()).is_cuda:
            model.to('cpu')
            self.update_used_vram()

    def update_used_vram(self):
        self.used_vram = self.calculate_total_used_vram()

    @torch.inference_mode()
    def safe_offload(self):
        """Safely offload all models to clear VRAM."""
        for model_id, (clip, vae, model_wrap) in self.loaded_models.items():
            self.offload_model(clip.cond_stage_model)
            self.offload_model(vae.first_stage_model)
            self.offload_model(model_wrap.model)
        self.update_used_vram()
    @torch.inference_mode()
    def load_model(self, clip_paths, vae_path, unet_path, list_em=True):
        model_id = (tuple(clip_paths), vae_path, unet_path)

        with self._lock:
            if model_id in self.loaded_models:
                return self.loaded_models[model_id]
            clip_type = sd.CLIPType.FLUX
            clip = sd.load_clip(ckpt_paths=clip_paths, embedding_directory=None, clip_type=clip_type)
            vae = sd.VAE(sd=utils.load_torch_file(vae_path))
            model = sd.load_unet(unet_path, dtype=torch.float8_e5m2)
            if list_em:
                self.loaded_models[model_id] = (clip, vae, model)
                self.update_used_vram()
            return clip, vae, model

    # @torch.inference_mode()
    # def load_model(self, clip_paths, vae_path, unet_path):
    #     model_id = (tuple(clip_paths), vae_path, unet_path)
    #     diff_file_path = unet_path.replace(".safetensors", "_diff.safetensors")
    #
    #     with self._lock:
    #         if model_id in self.loaded_models:
    #             return self.loaded_models[model_id]
    #
    #         clip_type = sd.CLIPType.FLUX
    #         clip = sd.load_clip(ckpt_paths=clip_paths, embedding_directory=None, clip_type=clip_type)
    #         vae = sd.VAE(sd=utils.load_torch_file(vae_path))
    #         model = sd.load_unet(unet_path, dtype=torch.float8_e5m2)
    #
    #         # # Check if the diff file exists, and apply it if needed
    #         # if os.path.exists(diff_file_path):
    #         #     apply_diff(model.model, diff_file_path, reverse=False)
    #         # else:
    #         #     # Check if a dev model is available, load it and save the difference
    #         #     dev_unet_path = unet_path.replace("-schnell", "-dev")
    #         #     if os.path.exists(dev_unet_path):
    #         #         dev_model = sd.load_unet(dev_unet_path, dtype=torch.float16)
    #         #         save_model_diff(dev_model.model, model.model, diff_file_path)
    #
    #         self.loaded_models[model_id] = (clip, vae, model)
    #         self.update_used_vram()
    #
    #         return clip, vae, model


model_manager = ModelManager()