import os
import platform

import torch
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

OP_NODE_TORCH_LOADER = get_next_opcode()


def encode(cls, pixel_samples):
    pixel_samples = pixel_samples.movedim(-1, 1)
    # try:
    #     memory_used = self.memory_used_encode(pixel_samples.shape, self.vae_dtype)
    #     model_management.load_models_gpu([self.patcher], memory_required=memory_used)
    #     free_memory = model_management.get_free_memory(self.device)
    #     batch_number = int(free_memory / memory_used)
    #     batch_number = max(1, batch_number)
    #     samples = torch.empty((pixel_samples.shape[0], self.latent_channels,
    #                            round(pixel_samples.shape[2] // self.downscale_ratio),
    #                            round(pixel_samples.shape[3] // self.downscale_ratio)), device=self.output_device)
    #     for x in range(0, pixel_samples.shape[0], batch_number):
    #         pixels_in = (2. * pixel_samples[x:x + batch_number] - 1.).to(self.vae_dtype).to(self.device)
    #         samples[x:x + batch_number] = self.first_stage_model.encode(pixels_in).to(self.output_device).float()
    #
    # except model_management.OOM_EXCEPTION as e:
    #     print("Warning: Ran out of memory when regular VAE encoding, retrying with tiled VAE encoding.")
    samples = cls.encode_tiled_(pixel_samples)

    return samples


def replace_fn(instance, new_function):
    """
    Replaces the 'encode' method of the instance with the new_function.

    Parameters:
    instance: The instance of the class whose 'encode' method is to be replaced.
    new_function: The new function to replace the 'encode' method.
    """
    # Bind the new function to the instance
    bound_function = new_function.__get__(instance, instance.__class__)

    # Replace the 'encode' method
    instance.encode = bound_function

def load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, output_clipvision=False, embedding_directory=None, output_model=True, *args, **kwargs):
    import comfy
    import torch
    from comfy import model_management
    from comfy import clip_vision
    from comfy import model_detection
    import comfy.taesd.taesd
    from comfy.sd import load_model_weights, VAE, CLIP
    load_path = gs.prefs.checkpoints + ckpt_path
    sd = comfy.utils.load_torch_file(load_path)
    sd_keys = sd.keys()
    clip = None
    clipvision = None
    vae = None
    model = None
    model_patcher = None
    clip_target = None

    parameters = comfy.utils.calculate_parameters(sd, "model.diffusion_model.")
    unet_dtype = model_management.unet_dtype(model_params=parameters)
    load_device = model_management.get_torch_device()
    load_device = gs.device
    manual_cast_dtype = model_management.unet_manual_cast(unet_dtype, load_device)

    class WeightsLoader(torch.nn.Module):
        pass

    model_config = model_detection.model_config_from_unet(sd, "model.diffusion_model.", unet_dtype)
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)

    if model_config is None:
        raise RuntimeError("ERROR: Could not detect model type of: {}".format(load_path))

    if model_config.clip_vision_prefix is not None:
        if output_clipvision:
            clipvision = clip_vision.load_clipvision_from_sd(sd, model_config.clip_vision_prefix, True)

    if output_model:
        inital_load_device = load_device
        offload_device = model_management.unet_offload_device()
        model = model_config.get_model(sd, "model.diffusion_model.", device=torch.device("cpu"))
        model.load_model_weights(sd, "model.diffusion_model.")
        model.to(manual_cast_dtype)

    if output_vae:
        vae_sd = comfy.utils.state_dict_prefix_replace(sd, {"first_stage_model.": ""}, filter_keys=True)
        vae_sd = model_config.process_vae_state_dict(vae_sd)
        gs.models[ckpt_path]["vae"] = VAE(sd=vae_sd, device=torch.device("cuda"), dtype=torch.bfloat16)
        gs.models[ckpt_path]["vae"].first_stage_model.bfloat16()
        gs.models[ckpt_path]["vae"].device = gs.device
        # gs.models[ckpt_path]["vae"].vae_dtype = torch.bfloat16


    # if output_clip:
    #     w = WeightsLoader()
    #     clip_target = model_config.clip_target()
    #     if clip_target is not None:
    #         gs.models[ckpt_path]["clip"] = CLIP(clip_target, embedding_directory=embedding_directory)
    #         w.cond_stage_model = gs.models[ckpt_path]["clip"].cond_stage_model
    #         sd = model_config.process_clip_state_dict(sd)
    #         load_model_weights(w, sd)

    if output_clip:
        clip_target = model_config.clip_target()
        if clip_target is not None:
            clip_sd = model_config.process_clip_state_dict(sd)
            if len(clip_sd) > 0:
                gs.models[ckpt_path]["clip"] = CLIP(clip_target, embedding_directory=embedding_directory)
                m, u = gs.models[ckpt_path]["clip"].load_sd(clip_sd, full_model=True)
                if len(m) > 0:
                    print("clip missing:", m)

                if len(u) > 0:
                    print("clip unexpected:", u)
            else:
                print("no CLIP/text encoder weights in checkpoint, the text encoder model will not be loaded.")


    left_over = sd.keys()
    if len(left_over) > 0:
        print("left over keys:", left_over)

    # if output_model:
    gs.models[ckpt_path]["model"] = comfy.model_patcher.ModelPatcher(model, load_device=gs.device, offload_device=torch.device("cpu"), current_device=torch.device("cpu"))
        # if inital_load_device != torch.device("cpu"):
        #     print("loaded straight to GPU")
        #     model_management.load_model_gpu(model_patcher)
    del sd
    del vae_sd
    # del w
    del model
    torch_gc()

    # model_patcher.model.to(gs.device)
    # model_patcher.load_device = gs.device
    def load_model():
        #print("Load HiJacked")
        pass
    def decode_tiled_(samples, tile_x=64, tile_y=64, overlap = 16):
        steps = samples.shape[0] * comfy.utils.get_tiled_scale_steps(samples.shape[3], samples.shape[2], tile_x, tile_y, overlap)
        steps += samples.shape[0] * comfy.utils.get_tiled_scale_steps(samples.shape[3], samples.shape[2], tile_x // 2, tile_y * 2, overlap)
        steps += samples.shape[0] * comfy.utils.get_tiled_scale_steps(samples.shape[3], samples.shape[2], tile_x * 2, tile_y // 2, overlap)
        pbar = comfy.utils.ProgressBar(steps)
        gs.models[ckpt_path]["vae"].first_stage_model.bfloat16()
        decode_fn = lambda a: (gs.models[ckpt_path]["vae"].first_stage_model.decode(a.to(gs.models[ckpt_path]["vae"].vae_dtype).to(gs.models[ckpt_path]["vae"].device)) + 1.0).float()
        output = torch.clamp((
            (comfy.utils.tiled_scale(samples, decode_fn, tile_x // 2, tile_y * 2, overlap, upscale_amount = gs.models[ckpt_path]["vae"].downscale_ratio, output_device=gs.models[ckpt_path]["vae"].output_device, pbar = pbar) +
            comfy.utils.tiled_scale(samples, decode_fn, tile_x * 2, tile_y // 2, overlap, upscale_amount = gs.models[ckpt_path]["vae"].downscale_ratio, output_device=gs.models[ckpt_path]["vae"].output_device, pbar = pbar) +
             comfy.utils.tiled_scale(samples, decode_fn, tile_x, tile_y, overlap, upscale_amount = gs.models[ckpt_path]["vae"].downscale_ratio, output_device=gs.models[ckpt_path]["vae"].output_device, pbar = pbar))
            / 3.0) / 2.0, min=0.0, max=1.0)
        return output
    def encode_tiled_(pixel_samples, tile_x=512, tile_y=512, overlap = 64):
        steps = pixel_samples.shape[0] * comfy.utils.get_tiled_scale_steps(pixel_samples.shape[3], pixel_samples.shape[2], tile_x, tile_y, overlap)
        steps += pixel_samples.shape[0] * comfy.utils.get_tiled_scale_steps(pixel_samples.shape[3], pixel_samples.shape[2], tile_x // 2, tile_y * 2, overlap)
        steps += pixel_samples.shape[0] * comfy.utils.get_tiled_scale_steps(pixel_samples.shape[3], pixel_samples.shape[2], tile_x * 2, tile_y // 2, overlap)
        pbar = comfy.utils.ProgressBar(steps)
        gs.models[ckpt_path]["vae"].first_stage_model.bfloat16()
        encode_fn = lambda a: gs.models[ckpt_path]["vae"].first_stage_model.encode((2. * a - 1.).to(gs.models[ckpt_path]["vae"].vae_dtype).to(gs.models[ckpt_path]["vae"].device)).float()
        samples = comfy.utils.tiled_scale(pixel_samples, encode_fn, tile_x, tile_y, overlap, upscale_amount = (1/gs.models[ckpt_path]["vae"].downscale_ratio), out_channels=gs.models[ckpt_path]["vae"].latent_channels, output_device=gs.models[ckpt_path]["vae"].output_device, pbar=pbar)
        samples += comfy.utils.tiled_scale(pixel_samples, encode_fn, tile_x * 2, tile_y // 2, overlap, upscale_amount = (1/gs.models[ckpt_path]["vae"].downscale_ratio), out_channels=gs.models[ckpt_path]["vae"].latent_channels, output_device=gs.models[ckpt_path]["vae"].output_device, pbar=pbar)
        samples += comfy.utils.tiled_scale(pixel_samples, encode_fn, tile_x // 2, tile_y * 2, overlap, upscale_amount = (1/gs.models[ckpt_path]["vae"].downscale_ratio), out_channels=gs.models[ckpt_path]["vae"].latent_channels, output_device=gs.models[ckpt_path]["vae"].output_device, pbar=pbar)
        samples /= 3.0
        return samples

    gs.models[ckpt_path]["clip"].load_model = load_model
    gs.models[ckpt_path]["vae"].load_model = load_model
    gs.models[ckpt_path]["vae"].decode_tiled_ = decode_tiled_
    gs.models[ckpt_path]["vae"].encode_tiled_ = encode_tiled_
    #gs.models[ckpt_path]["vae"].encode = replace_fn(gs.models[ckpt_path]["vae"], encode)
    gs.models[ckpt_path]["model"].load_model = load_model

    # if gs.vram_state != "low":
    #     gs.models[ckpt_path]["vae"].first_stage_model.bfloat16().cuda()
    #     gs.models[ckpt_path]["clip"].cond_stage_model.half().cuda()
    #     gs.models[ckpt_path]["model"].model.half().cuda()
    # else:
    gs.models[ckpt_path]["vae"].first_stage_model.bfloat16().cpu()
    gs.models[ckpt_path]["clip"].cond_stage_model.half().cpu()
    gs.models[ckpt_path]["model"].model.half().cpu()
    #torch_gc()
    return


class TorchLoaderWidget(QDMNodeContentWidget):
    def initUI(self):
        self.create_widgets()
        self.create_main_layout(grid=1)

    def create_widgets(self):
        checkpoint_folder = gs.prefs.checkpoints

        os.makedirs(checkpoint_folder, exist_ok=True)
        print(checkpoint_folder)
        checkpoint_files = []
        for root, dirs, files in os.walk(checkpoint_folder):
            for f in files:
                if f.endswith(('.ckpt', '.pt', '.bin', '.pth', '.safetensors')):
                    full_path = os.path.join(root, f)
                    checkpoint_files.append(full_path.replace(checkpoint_folder, ""))
        #checkpoint_files = [f for f in os.listdir(checkpoint_folder) if f.endswith(('.ckpt', '.pt', '.bin', '.pth', '.safetensors'))]
        self.dropdown = self.create_combo_box(checkpoint_files, "Model:")
        if checkpoint_files == []:
            self.dropdown.addItem("Please place a model in models/checkpoints")
            print(f"TORCH LOADER NODE: No model file found at {os.getcwd()}/models/checkpoints,")
            print(f"TORCH LOADER NODE: please download your favorite ckpt before Evaluating this node.")

        # config_folder = "models/configs"
        # config_files = [f for f in os.listdir(config_folder) if f.endswith((".yaml"))]
        # config_files = sorted(config_files, key=str.lower)
        # self.config_dropdown = self.create_combo_box(config_files, "Config:")
        # self.config_dropdown.setCurrentText("v1-inference_fp16.yaml")

        vae_folder = gs.prefs.vae

        os.makedirs(vae_folder, exist_ok=True)

        vae_files = [f for f in os.listdir(vae_folder) if f.endswith(('.ckpt', '.pt', '.bin', '.pth', '.safetensors'))]
        vae_files = sorted(vae_files, key=str.lower)
        self.vae_dropdown = self.create_combo_box(vae_files, "Vae")
        self.vae_dropdown.addItem("default")
        self.vae_dropdown.setCurrentText("default")
        #self.optimization = self.create_combo_box(["None"] + valid_optimizations, "LDM Optimization")

        self.force_reload = self.create_check_box("Force Reload")



class CenterExpandingSizePolicy(QtWidgets.QSizePolicy):
    def __init__(self, parent=None):
        super().__init__(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.parent = parent
        self.setHorizontalStretch(0)
        self.setVerticalStretch(0)
        self.setRetainSizeWhenHidden(True)
        self.setHorizontalPolicy(QtWidgets.QSizePolicy.Expanding)
        self.setVerticalPolicy(QtWidgets.QSizePolicy.Expanding)


@register_node(OP_NODE_TORCH_LOADER)
class TorchLoaderNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/torch.png"
    op_code = OP_NODE_TORCH_LOADER
    op_title = "Torch Loader"
    content_label_objname = "torch_loader_node"
    category = "base/loaders"
    input_socket_name = ["EXEC"]
    # output_socket_name = ["EXEC"]
    custom_output_socket_name = ["VAE", "CLIP", "MODEL", "EXEC"]
    def __init__(self, scene):
        super().__init__(scene, inputs=[1], outputs=[4,4,4,1])
        self.loader = ModelLoader()

    def initInnerClasses(self):
        self.content = TorchLoaderWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.icon = self.icon
        self.grNode.thumbnail = QtGui.QImage(self.grNode.icon).scaled(64, 64, QtCore.Qt.KeepAspectRatio)

        self.grNode.width = 800
        self.grNode.height = 250
        self.content.setMinimumHeight(140)
        self.content.setMinimumWidth(800)
        self.content.eval_signal.connect(self.evalImplementation)

        self.model = None
        self.clip = None
        self.vae = None
        self.loaded_vae = None
        self.loaded_sd = ""
    def remove(self):
        self.clean_sd()
        super().remove()
    def clean_sd(self):

        try:
            self.model.model.cpu()
            self.clip.cpu()
            self.vae.cpu()
        except:
            pass
        del self.model
        del self.clip
        del self.vae

        self.model = None
        self.clip = None
        self.vae = None

        torch_gc(full=True)

    def setOutput(self, index, value):
        """
        Set the value of the output socket with the given index.

        Args:
            index (int): The index of the output socket.
            value: The value to be set for the output socket.
        """
        pass
        # object_name = self.getID(index)
        # self._output_values[object_name] = value  # Store the reference in the dictionary

    def getOutput(self, index=0, origin_index=0):
        """
         Get the value of the output socket with the given index.

         Args:
             index (int): The index of the output socket.

         Returns:
             The value of the output socket, or None if it does not exist.
         """

        if self.loaded_sd in gs.models:
            if index == 0:
                return gs.models[self.loaded_sd]["vae"]
            elif index == 1:
                return gs.models[self.loaded_sd]["clip"]
            elif index == 2:
                return gs.models[self.loaded_sd]["model"]
        else:
            return None
        # object_name = self.getID(index)
        # return self._output_values.get(object_name, None)  # Get the value using the dictionary
    def evalImplementation_thread(self, index=0):
        self.busy = True
        model_name = self.content.dropdown.currentText()
        inpaint = True if "inpaint" in model_name else False
        m = "sd_model" if not inpaint else "inpaint"

        if model_name not in gs.models:

            gs.models[model_name] = {}
            # gs.models[model_name]["model"], gs.models[model_name]["clip"], gs.models[model_name]["vae"], _ = load_checkpoint_guess_config(os.path.join(gs.prefs.checkpoints, model_name))
            load_checkpoint_guess_config(model_name)
            self.loaded_sd = model_name
            torch_gc(full=True)
            self.scene.getView().parent().window().update_models_signal.emit()

        else:
            self.loaded_sd = model_name

            #torch_gc()
        if self.content.vae_dropdown.currentText() != 'default':
            model = self.content.vae_dropdown.currentText()
            gs.models[self.loaded_sd]["vae"] = self.load_vae(model)
            self.loaded_vae = model
        else:
            self.loaded_vae = 'default'
        if self.loaded_vae != self.content.vae_dropdown.currentText():
            model = self.content.vae_dropdown.currentText()
            gs.models[self.loaded_sd]["vae"] = self.load_vae(model)
            self.loaded_vae = model
        return [None, None, None, None]

    def load_vae(self, file):
        from comfy.sd import VAE
        import comfy
        path = os.path.join(gs.prefs.vae, file)
        print("Loading", path)
        # gs.models["sd"].first_stage_model.cpu()
        # gs.models["sd"].first_stage_model = None
        # gs.models["sd"].first_stage_model = VAE(ckpt_path=path)
        sd = comfy.utils.load_torch_file(path)
        vae = VAE(sd=sd)
        vae.first_stage_model.bfloat16().cuda()
        vae.vae_dtype = torch.bfloat16
        print("VAE Loaded", file)
        del sd
        torch_gc()
        return vae
        # if self.loaded_sd != model_name or self.content.force_reload.isChecked() == True:
        #     self.clean_sd()
        #     self.model, self.clip, self.vae, self.clipvision = self.loader.load_checkpoint_guess_config(model_name, style="None")
        #     self.loaded_sd = model_name
        # if self.content.vae_dropdown.currentText() != 'default':
        #     model = self.content.vae_dropdown.currentText()
        #     self.vae = self.loader.load_vae(model)
        #     self.loaded_vae = model
        # else:
        #     self.loaded_vae = 'default'
        # if self.loaded_vae != self.content.vae_dropdown.currentText():
        #     model = self.content.vae_dropdown.currentText()
        #     self.vae = self.loader.load_vae(model)
        #     self.loaded_vae = model
        # return [self.vae, self.clip, self.model]



