import secrets

import numpy as np
import torch
from PIL import Image
from qtpy import QtWidgets, QtCore, QtGui

from backend_helpers.torch_helpers.resizeRight import resizeright, interp_methods

from ainodes_frontend import singleton as gs
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend.node_engine.utils import dumpException

OP_NODE_LATENT = get_next_opcode()
OP_NODE_LATENT_COMPOSITE = get_next_opcode()
class LatentWidget(QDMNodeContentWidget):
    def initUI(self):
        # Create a label to display the image
        self.create_widgets()
        self.create_main_layout(grid=1)

    def create_widgets(self):
        self.width = self.create_spin_box("Width", 64, 4096, 512, 64)
        self.height = self.create_spin_box("Height", 64, 4096, 512, 64)
        self.rescale_latent = self.create_check_box("Latent Rescale")

        self.noise_seed = self.create_line_edit("Noise Seed")
        self.noise_subseed = self.create_line_edit("Noise Subseed")
        self.use_subnoise = self.create_check_box("Use Subnoise")
        self.subnoise_width = self.create_spin_box("Subnoise Width", 64, 4096, 512, 64)
        self.subnoise_height = self.create_spin_box("Subnoise Height", 64, 4096, 512, 64)
        self.subnoise_strength = self.create_double_spin_box("Subnoise strength", min_val=0.0, max_val=10.0, default_val=1.0)

        self.force_encode = self.create_check_box("Force Encode")

@register_node(OP_NODE_LATENT)
class LatentNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/empty_latent.png"
    op_code = OP_NODE_LATENT
    op_title = "Empty Latent Image"
    content_label_objname = "empty_latent_node"
    category = "base/torch"
    custom_input_socket_name = ["VAE", "LATENT", "IMAGE", "EXEC"]

    make_dirty = True
    dim = (340, 600)
    NodeContent_class = LatentWidget

    #force_run = True

    def __init__(self, scene):

        super().__init__(scene, inputs=[4,2,5,1], outputs=[2,1])
        #self.eval()

    # def initInnerClasses(self):
    #     self.content = LatentWidget(self)
    #     self.grNode = CalcGraphicsNode(self)
    #     self.grNode.icon = self.icon
    #     self.grNode.thumbnail = QtGui.QImage(self.grNode.icon).scaled(64, 64, QtCore.Qt.KeepAspectRatio)
    #
    #     self.input_socket_name = ["EXEC", "IMAGE", "LATENT"]
    #     self.output_socket_name = ["EXEC", "LATENT"]
    #     self.grNode.height = 400
    #     self.grNode.width = 200
    #     self.content.eval_signal.connect(self.evalImplementation)

    #@QtCore.Slot()
    def evalImplementation_thread(self, index=0):
        vae = self.getInputData(0)
        add_mask = False
        # if self.getInput(1) != None:
        #     latent_node, index = self.getInput(1)
        samples = self.getInputData(1)
        input_image = self.getInputData(2)

        # if samples == None and self.content.force_encode.isChecked():
        #     return [None]

        if samples is not None:
            if "noise_mask" in samples:
                add_mask = True
                noise_mask = samples["noise_mask"]


            samples = samples["samples"]
            if self.content.force_encode.isChecked():
                with torch.inference_mode():
                    vae.first_stage_model.cuda()
                    samples = vae.encode_tiled_(vae, samples)
                    vae.first_stage_model.cpu()
            print(f"[ Using Input Latent: {samples.shape} ]")
            # except:
            #     print(f"EMPTY LATENT NODE: Tried using Latent input, but found an invalid value, generating latent with parameters: {self.content.width.value(), self.content.height.value()}")
            #     samples = self.generate_latent()
        elif input_image is not None:
            vae.first_stage_model.cuda()
            input_image = input_image.movedim(-1, 1).detach().clone()
            with torch.inference_mode():
                samples = vae.encode_tiled_(input_image)
            vae.first_stage_model.cpu()
        else:
            samples = self.generate_latent()
        if self.content.rescale_latent.isChecked() == True:
            rescaled_samples = []
            for sample in samples:
                #sample = sample.detach().float()
                return_sample = resizeright.resize(sample, scale_factors=None,
                                                out_shape=[sample.shape[0], sample.shape[1], int(self.content.height.value() // 8),
                                                        int(self.content.width.value() // 8)],
                                                interp_method=interp_methods.lanczos3, support_sz=None,
                                                antialiasing=True, by_convs=True, scale_tolerance=None,
                                                max_numerator=10, pad_mode='reflect')#.half()

                rescaled_samples.append(return_sample)
            samples = rescaled_samples
            if gs.logging:
                print(f"{len(samples)}x Latents rescaled to: {samples[0].shape}")
        #print(samples[0].shape)
        result = {"samples":samples}
        if add_mask:
            result["noise_mask"] = noise_mask
        #print("will return", samples.shape)
        return [result]
            #return self.value

    # ##@QtCore.Slot(object)
    # def onWorkerFinished(self, result, exec=True):
    #     self.busy = False
    #     #super().onWorkerFinished(None)
    #     self.markDirty(False)
    #     self.markInvalid(False)
    #     self.setOutput(0, result)
    #     self.content.update()
    #
    #     self.content.finished.emit()
    #     if exec:
    #         if len(self.getOutputs(1)) > 0:
    #             self.executeChild(output_index=1)
    # def onMarkedDirty(self):
    #     self.value = None
    def encode_image(self, init_image=None):
        init_latent = gs.models["vae"].encode(init_image)
        latent = init_latent.detach().to("cpu")# move to latent space
        del init_latent
        #torch_gc()
        return latent

    def generate_latent(self):
        width = self.content.width.value()
        height = self.content.height.value()
        seed = self.content.noise_seed.text()
        subseed = self.content.noise_subseed.text()
        try:
            seed = int(seed)
        except:
            seed = ""
        seed = seed if seed != "" else secrets.randbelow(9999999999)
        target_shape = (4, height // 8, width // 8)
        subheight = self.content.subnoise_height.value()
        subwidth = self.content.subnoise_width.value()
        noise_shape = target_shape if not self.content.use_subnoise.isChecked() else (target_shape[0], subheight // 8, subwidth // 8)
        subnoise = None

        if self.content.use_subnoise.isChecked():
            try:
                subseed = int(subseed)
            except:
                subseed = ""
            subseed = subseed if subseed != "" else secrets.randbelow(9999999999)

            torch.manual_seed(subseed)
            subnoise = torch.randn(noise_shape, device=gs.device.type)
        torch.manual_seed(seed)
        noise = torch.randn(noise_shape, device=gs.device.type)


        if subnoise is not None:
            subnoise_strength = self.content.subnoise_strength.value()
            noise = slerp(subnoise_strength, noise, subnoise)

        if noise_shape != target_shape:
            torch.manual_seed(seed)
            x = torch.randn(target_shape, device=gs.device.type)
            dx = (target_shape[2] - noise_shape[2]) // 2
            dy = (target_shape[1] - noise_shape[1]) // 2
            w = noise_shape[2] if dx >= 0 else noise_shape[2] + 2 * dx
            h = noise_shape[1] if dy >= 0 else noise_shape[1] + 2 * dy
            tx = 0 if dx < 0 else dx
            ty = 0 if dy < 0 else dy
            dx = max(-dx, 0)
            dy = max(-dy, 0)

            x[:, ty:ty+h, tx:tx+w] = noise[:, dy:dy+h, dx:dx+w]
            noise = x

        return noise[None]

class LatentCompositeWidget(QDMNodeContentWidget):
    def initUI(self):
        # Create a label to display the image
        self.width = QtWidgets.QSpinBox()
        self.width.setMinimum(64)
        self.width.setMaximum(4096)
        self.width.setValue(64)
        self.width.setSingleStep(64)

        self.height = QtWidgets.QSpinBox()
        self.height.setMinimum(64)
        self.height.setMaximum(4096)
        self.height.setValue(64)
        self.height.setSingleStep(64)

        self.feather = QtWidgets.QSpinBox()
        self.feather.setMinimum(0)
        self.feather.setMaximum(200)
        self.feather.setValue(10)
        self.feather.setSingleStep(1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(15,15,15,20)
        layout.setSpacing(10)
        layout.addWidget(self.width)
        layout.addWidget(self.height)
        layout.addWidget(self.feather)
        self.setLayout(layout)

    def serialize(self):
        res = super().serialize()
        #res['value'] = self.edit.text()
        return res

    def deserialize(self, data, hashmap={}):
        res = super().deserialize(data, hashmap)
        try:
            value = data['value']
            #self.image.setPixmap(value)
            return True & res
        except Exception as e:
            dumpException(e)
        return res


@register_node(OP_NODE_LATENT_COMPOSITE)
class LatentCompositeNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/comp_latent.png"
    op_code = OP_NODE_LATENT_COMPOSITE
    op_title = "Composite Latent Images"
    content_label_objname = "latent_comp_node"
    category = "base/torch"
    NodeContent_class = LatentCompositeWidget
    dim = (340, 340)

    def __init__(self, scene):
        super().__init__(scene, inputs=[2,2,1], outputs=[2,1])
    # def initInnerClasses(self):
    #     self.content = LatentCompositeWidget(self)
    #     self.grNode = CalcGraphicsNode(self)
    #     self.input_socket_name = ["EXEC", "LATENT1", "LATENT2"]
    #     self.output_socket_name = ["EXEC", "LATENT"]
    #     self.grNode.height = 220
    #     self.grNode.width = 240
    #     self.grNode.icon = self.icon
    #     self.content.eval_signal.connect(self.evalImplementation)

    def evalImplementation_thread(self, index=0):

        if self.isDirty() == True:
            if self.getInput(index) != None:

                self.value = self.composite()
        else:
            return [self.value]


    def onMarkedDirty(self):
        self.value = None

    def composite(self):
        width = self.content.width.value()
        height = self.content.height.value()
        feather = self.content.feather.value()
        x =  width // 8
        y = height // 8
        feather = feather // 8
        # samples_out = self.getInput(0)
        s = self.getInput(0)
        samples_to = self.getInput(0)
        samples_from = self.getInput(1)

        if samples_to is not None and samples_from is not None:
            samples_to = samples_to["samples"]
            samples_from = samples_from["samples"]

        if feather == 0:
            s[:,:,y:y+samples_from.shape[2],x:x+samples_from.shape[3]] = samples_from[:,:,:samples_to.shape[2] - y, :samples_to.shape[3] - x]
        else:
            samples_from = samples_from[:,:,:samples_to.shape[2] - y, :samples_to.shape[3] - x]
            mask = torch.ones_like(samples_from)
            for t in range(feather):
                if y != 0:
                    mask[:,:,t:1+t,:] *= ((1.0/feather) * (t + 1))

                if y + samples_from.shape[2] < samples_to.shape[2]:
                    mask[:,:,mask.shape[2] -1 -t: mask.shape[2]-t,:] *= ((1.0/feather) * (t + 1))
                if x != 0:
                    mask[:,:,:,t:1+t] *= ((1.0/feather) * (t + 1))
                if x + samples_from.shape[3] < samples_to.shape[3]:
                    mask[:,:,:,mask.shape[3]- 1 - t: mask.shape[3]- t] *= ((1.0/feather) * (t + 1))
            rev_mask = torch.ones_like(mask) - mask
            s[:,:,y:y+samples_from.shape[2],x:x+samples_from.shape[3]] = samples_from[:,:,:samples_to.shape[2] - y, :samples_to.shape[3] - x] * mask + s[:,:,y:y+samples_from.shape[2],x:x+samples_from.shape[3]] * rev_mask

        #self.setOutput(0, s)
        return {"samples":s}
def load_img(image, shape=None, use_alpha_as_mask=False):
    # use_alpha_as_mask: Read the alpha channel of the image as the mask image
    #if path.startswith('http://') or path.startswith('https://'):
    #    image = Image.open(requests.get(path, stream=True).raw)
    #else:
    #    image = Image.open(path)

    if use_alpha_as_mask:
        image = image.convert('RGBA')
    else:
        image = image.convert('RGB')

    if shape is not None:
        image = image.resize(shape, resample=Image.Resampling.LANCZOS)

    mask_image = None
    if use_alpha_as_mask:
        # Split alpha channel into a mask_image
        red, green, blue, alpha = Image.Image.split(image)
        mask_image = alpha.convert('L')
        image = image.convert('RGB')

    image = np.array(image).astype(np.float16) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    image = 2. * image - 1.

    return image, mask_image

# from https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475/3
def slerp(val, low, high):
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    dot = (low_norm*high_norm).sum(1)

    if dot.mean() > 0.9995:
        return low * val + high * (1 - val)

    omega = torch.acos(dot)
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res