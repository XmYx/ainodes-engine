import contextlib

import torch

from ainodes_backend import cldm
from ainodes_backend import singleton as gs

def load_controlnet(ckpt_path, model=None):
    controlnet_data = load_torch_file(ckpt_path)
    pth_key = 'control_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight'
    pth = False
    sd2 = False
    key = 'input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight'
    if pth_key in controlnet_data:
        pth = True
        key = pth_key
    elif key in controlnet_data:
        pass
    else:
        print("error checkpoint does not contain controlnet data", ckpt_path)
        return None

    context_dim = controlnet_data[key].shape[1]

    use_fp16 = False
    if controlnet_data[key].dtype == torch.float16:
        use_fp16 = True

    control_model = cldm.ControlNet(image_size=32,
                                    in_channels=4,
                                    hint_channels=3,
                                    model_channels=320,
                                    attention_resolutions=[ 4, 2, 1 ],
                                    num_res_blocks=2,
                                    channel_mult=[ 1, 2, 4, 4 ],
                                    num_heads=8,
                                    use_spatial_transformer=True,
                                    transformer_depth=1,
                                    context_dim=context_dim,
                                    use_checkpoint=True,
                                    legacy=False,
                                    use_fp16=use_fp16)

    if pth:
        if 'difference' in controlnet_data:
            print(model)
            if model is not None:
                m = model.patch_model()
                model_sd = m.state_dict()
                for x in controlnet_data:
                    c_m = "control_model."
                    if x.startswith(c_m):
                        sd_key = "model.diffusion_model.{}".format(x[len(c_m):])
                        if sd_key in model_sd:
                            cd = controlnet_data[x]
                            cd += model_sd[sd_key].type(cd.dtype).to(cd.device)
                model.unpatch_model()
            else:
                print("WARNING: Loaded a diff controlnet without a model. It will very likely not work.")

        class WeightsLoader(torch.nn.Module):
            pass
        w = WeightsLoader()
        w.control_model = control_model
        w.load_state_dict(controlnet_data, strict=False)
    else:
        control_model.load_state_dict(controlnet_data, strict=False)


    gs.models["controlnet"] = ControlNet(control_model)
    del controlnet_data
    del control_model
    #return control

class ControlNet:
    def __init__(self, control_model, device="cuda"):
        self.control_model = control_model
        self.cond_hint_original = None
        self.cond_hint = None
        self.strength = 1.0
        self.device = device
        self.previous_controlnet = None

    def get_control(self, x_noisy, t, cond_txt, batched_number):
        control_prev = None
        if self.previous_controlnet is not None:
            control_prev = self.previous_controlnet.get_control(x_noisy, t, cond_txt, batched_number)

        output_dtype = x_noisy.dtype
        if self.cond_hint is None or x_noisy.shape[2] * 8 != self.cond_hint.shape[2] or x_noisy.shape[3] * 8 != self.cond_hint.shape[3]:
            if self.cond_hint is not None:
                del self.cond_hint
            self.cond_hint = None
            self.cond_hint = resize_image_to(self.cond_hint_original, x_noisy, batched_number).to(self.control_model.dtype).to(self.device)

        if self.control_model.dtype == torch.float16:
            precision_scope = torch.autocast
        else:
            precision_scope = contextlib.nullcontext

        with precision_scope(self.device):
            #self.control_model = model_management.load_if_low_vram(self.control_model)
            control = self.control_model(x=x_noisy, hint=self.cond_hint, timesteps=t, context=cond_txt)
            #self.control_model = model_management.unload_if_low_vram(self.control_model)
        out = {'middle':[], 'output': []}
        autocast_enabled = torch.is_autocast_enabled()

        for i in range(len(control)):
            if i == (len(control) - 1):
                key = 'middle'
                index = 0
            else:
                key = 'output'
                index = i
            x = control[i]
            x *= self.strength
            if x.dtype != output_dtype and not autocast_enabled:
                x = x.to(output_dtype)

            if control_prev is not None and key in control_prev:
                prev = control_prev[key][index]
                if prev is not None:
                    x += prev
            out[key].append(x)
        if control_prev is not None and 'input' in control_prev:
            out['input'] = control_prev['input']
        return out

    def set_cond_hint(self, cond_hint, strength=1.0):
        self.cond_hint_original = cond_hint
        self.strength = strength
        return self

    def set_previous_controlnet(self, controlnet):
        self.previous_controlnet = controlnet
        return self

    def cleanup(self):
        if self.previous_controlnet is not None:
            self.previous_controlnet.cleanup()
        if self.cond_hint is not None:
            del self.cond_hint
            self.cond_hint = None

    def copy(self):
        c = ControlNet(self.control_model)
        c.cond_hint_original = self.cond_hint_original
        c.strength = self.strength
        return c

    def get_control_models(self):
        out = []
        if self.previous_controlnet is not None:
            out += self.previous_controlnet.get_control_models()
        out.append(self.control_model)
        return out
def load_torch_file(ckpt):
    if ckpt.lower().endswith(".safetensors"):
        import safetensors.torch
        sd = safetensors.torch.load_file(ckpt, device="cpu")
    else:
        pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        if "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
        else:
            sd = pl_sd
    return sd
def resize_image_to(tensor, target_latent_tensor, batched_number):
    tensor = common_upscale(tensor, target_latent_tensor.shape[3] * 8, target_latent_tensor.shape[2] * 8, 'nearest-exact', "center")
    target_batch_size = target_latent_tensor.shape[0]

    current_batch_size = tensor.shape[0]
    #print(current_batch_size, target_batch_size)
    if current_batch_size == 1:
        return tensor

    per_batch = target_batch_size // batched_number
    tensor = tensor[:per_batch]

    if per_batch > tensor.shape[0]:
        tensor = torch.cat([tensor] * (per_batch // tensor.shape[0]) + [tensor[:(per_batch % tensor.shape[0])]], dim=0)

    current_batch_size = tensor.shape[0]
    if current_batch_size == target_batch_size:
        return tensor
    else:
        return torch.cat([tensor] * batched_number, dim=0)

def common_upscale(samples, width, height, upscale_method, crop):
    if crop == "center":
        #print("RESIZE", samples.shape)
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
    return torch.nn.functional.interpolate(s, size=(height, width), mode=upscale_method)
