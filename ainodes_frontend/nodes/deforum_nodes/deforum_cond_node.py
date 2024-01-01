import contextlib
import math

import numpy as np
import torch

from ainodes_frontend.base.qimage_ops import tensor2pil, pil2tensor
# from ai_nodes.ainodes_engine_base_nodes.image_nodes.image_op_node import HWC3
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend import singleton as gs
OP_NODE_DEFORUM_SD_COND = get_next_opcode()
import torch.nn.functional as F

def pyramid_blend(tensor1, tensor2, blend_value):
    # For simplicity, we'll use two levels of blending
    downsampled1 = F.avg_pool2d(tensor1, 2)
    downsampled2 = F.avg_pool2d(tensor2, 2)

    blended_low = (1 - blend_value) * downsampled1 + blend_value * downsampled2
    blended_high = tensor1 + tensor2 - F.interpolate(blended_low, scale_factor=2)

    return blended_high
def gaussian_blend(tensor2, tensor1, blend_value):
    sigma = 0.5  # Adjust for desired smoothness
    weight = math.exp(-((blend_value - 0.5) ** 2) / (2 * sigma ** 2))
    return (1 - weight) * tensor1 + weight * tensor2
def sigmoidal_blend(tensor1, tensor2, blend_value):
    # Convert blend_value into a tensor with the same shape as tensor1 and tensor2
    blend_tensor = torch.full_like(tensor1, blend_value)
    weight = 1 / (1 + torch.exp(-10 * (blend_tensor - 0.5)))  # Sigmoid function centered at 0.5
    return (1 - weight) * tensor1 + weight * tensor2


blend_methods = ["linear", "sigmoidal", "gaussian", "pyramid"]

def blend_tensors(obj1, obj2, blend_value, blend_method="linear"):
    """
    Blends tensors in two given objects based on a blend value using various blending strategies.
    """

    if blend_method == "linear":
        weight = blend_value
        blended_cond = (1 - weight) * obj1[0] + weight * obj2[0]
        blended_pooled = (1 - weight) * obj1[1]['pooled_output'] + weight * obj2[1]['pooled_output']

    elif blend_method == "sigmoidal":
        blended_cond = sigmoidal_blend(obj1[0], obj2[0], blend_value)
        blended_pooled = sigmoidal_blend(obj1[1]['pooled_output'], obj2[1]['pooled_output'], blend_value)

    elif blend_method == "gaussian":
        blended_cond = gaussian_blend(obj1[0], obj2[0], blend_value)
        blended_pooled = gaussian_blend(obj1[1]['pooled_output'], obj2[1]['pooled_output'], blend_value)

    elif blend_method == "pyramid":
        blended_cond = pyramid_blend(obj1[0], obj2[0], blend_value)
        blended_pooled = pyramid_blend(obj1[1]['pooled_output'], obj2[1]['pooled_output'], blend_value)

    return [[blended_cond, {"pooled_output": blended_pooled}]]

class DeforumConditioningWidget(QDMNodeContentWidget):
    def initUI(self):
        self.blend_method = self.create_combo_box(blend_methods, "Blend Method")
        self.create_main_layout(grid=1)


@register_node(OP_NODE_DEFORUM_SD_COND)
class DeforumConditioningNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/conditioning.png"
    op_code = OP_NODE_DEFORUM_SD_COND
    op_title = "Deforum SD Conditioning"
    content_label_objname = "deforum_sdcond_node"
    category = "base/deforum"
    NodeContent_class = DeforumConditioningWidget
    dim = (240, 140)

    make_dirty = True
    custom_input_socket_name = ["CLIP", "DATA", "IMAGE", "CONTROL_NET", "EXEC"]
    custom_output_socket_name = ["DATA", "NEGATIVE", "POSITIVE", "EXEC"]


    def __init__(self, scene):
        super().__init__(scene, inputs=[4,6,5,4,1], outputs=[6,3,3,1])
        # Create a worker object
        self.device = gs.device
        if self.device in [torch.device('mps'), torch.device('cpu')]:
            self.context = contextlib.nullcontext()
        else:
            self.context = torch.autocast(gs.device.type)
        self.clip_skip = -2

    #@QtCore.Slot()
    def evalImplementation_thread(self, index=0, prompt_override=None):
        clip = self.getInputData(0)

        data = self.getInputData(1)
        if data is not None and gs.should_run:

            prompt = data.get("prompt", "")
            negative_prompt = data.get("negative_prompt", "")

            next_prompt = data.get("next_prompt", None)

            print(f"[ Deforum Conds: {prompt}, {negative_prompt} ]")
            cond = self.get_conditioning(prompt=prompt, clip=clip)
            image = self.getInputData(2)
            controlnet = self.getInputData(3)
            if image is not None:
                if controlnet is not None:
                    cnet_image = self.get_canny_image(image)
                    with torch.inference_mode():
                        print("[ Applying controlnet - Loopback Mode] ")
                        cond = self.apply_controlnet(cond, controlnet, cnet_image, 0.7)
            prompt_blend = data.get("prompt_blend", 0.0)
            if next_prompt != prompt and prompt_blend != 0.0 and next_prompt is not None:
                next_cond = self.get_conditioning(prompt=next_prompt, clip=clip)
                with torch.inference_mode():
                    cond = blend_tensors(cond[0], next_cond[0], prompt_blend, self.content.blend_method.currentText())
                print(f"[ Deforum Cond Blend: {next_prompt}, {prompt_blend} ]")

            n_cond = self.get_conditioning(prompt=negative_prompt, clip=clip)






            return [data, n_cond, cond]
        else:
            return [None, None, None]
    def get_canny_image(self, image):
        image = tensor2pil(image)
        image = np.array(image)
        import cv2
        from PIL import Image
        image = cv2.Canny(image, 0, 200, L2gradient=True)
        image = HWC3(image)
        image = Image.fromarray(image)
        image = pil2tensor(image)
        return image
    def apply_controlnet(self, conditioning, control_net, image, strength):
        if strength == 0:
            return (conditioning, )

        c = []
        control_hint = image.movedim(-1,1)
        for t in conditioning:
            n = [t[0], t[1].copy()]
            c_net = control_net.copy().set_cond_hint(control_hint, strength)
            if 'control' in t[1]:
                c_net.set_previous_controlnet(t[1]['control'])
            n[1]['control'] = c_net
            n[1]['control_apply_to_uncond'] = True
            c.append(n)
        return c
    def get_conditioning(self, prompt="", clip=None, progress_callback=None):

        """if gs.loaded_models["loaded"] == []:
            for node in self.scene.nodes:
                if isinstance(node, TorchLoaderNode):
                    node.evalImplementation()
                    #print("Node found")"""



        with self.context:
            with torch.no_grad():
                clip_skip = -2
                if self.clip_skip != clip_skip or clip.layer_idx != clip_skip:
                    clip.layer_idx = clip_skip
                    clip.clip_layer(clip_skip)
                    self.clip_skip = clip_skip

                tokens = clip.tokenize(prompt)
                cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)

                return [[cond, {"pooled_output": pooled}]]
