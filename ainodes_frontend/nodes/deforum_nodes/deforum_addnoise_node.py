
import cv2
import numpy as np
import torch
from PIL import Image

from ainodes_frontend.base.qimage_ops import pil2tensor
from deforum.generators.deforum_noise_generator import add_noise
from deforum.utils.image_utils import unsharp_mask, compose_mask_with_check

# from deforum.avfunctions.image.image_sharpening import unsharp_mask
# from deforum.avfunctions.masks.composable_masks import compose_mask_with_check
# from deforum.avfunctions.noise.noise import add_noise
# from ai_nodes.ainodes_engine_base_nodes.ainodes_backend import pil2tensor
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget

from ainodes_frontend import singleton as gs
OP_NODE_DEFORUM_ADDNOISE = get_next_opcode()

class DeforumAddNoiseWidget(QDMNodeContentWidget):
    def initUI(self):
        self.create_main_layout(grid=1)


@register_node(OP_NODE_DEFORUM_ADDNOISE)
class DeforumAddNoiseNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/conditioning.png"
    op_code = OP_NODE_DEFORUM_ADDNOISE
    op_title = "Deforum Add Noise"
    content_label_objname = "deforum_addnoise_node"
    category = "Deforum"
    NodeContent_class = DeforumAddNoiseWidget
    dim = (240, 140)

    make_dirty = True
    custom_input_socket_name = ["DATA", "IMAGE", "EXEC"]
    custom_output_socket_name = ["DATA", "IMAGE", "EXEC"]


    def __init__(self, scene):
        super().__init__(scene, inputs=[6,5,1], outputs=[6,5,1])

    #@QtCore.Slot()
    def evalImplementation_thread(self, index=0, prompt_override=None):

        data = self.getInputData(0)
        image = self.getInputData(1)

        if image is not None:

            keys = data.get("keys")
            args = data.get("args")
            anim_args = data.get("anim_args")
            root = data.get("root")
            frame_idx = data.get("frame_idx")
            noise = keys.noise_schedule_series[frame_idx]
            kernel = int(keys.kernel_schedule_series[frame_idx])
            sigma = keys.sigma_schedule_series[frame_idx]
            amount = keys.amount_schedule_series[frame_idx]
            threshold = keys.threshold_schedule_series[frame_idx]
            contrast = keys.contrast_schedule_series[frame_idx]
            if anim_args.use_noise_mask and keys.noise_mask_schedule_series[frame_idx] is not None:
                noise_mask_seq = keys.noise_mask_schedule_series[frame_idx]
            else:
                noise_mask_seq = None
            mask_vals = {}
            noise_mask_vals = {}

            mask_vals['everywhere'] = Image.new('1', (args.W, args.H), 1)
            noise_mask_vals['everywhere'] = Image.new('1', (args.W, args.H), 1)

            from ai_nodes.ainodes_engine_deforum_nodes.deforum_pipeline_nodes.deforum_framewarp_node import tensor2np
            prev_img = tensor2np(image)
            mask_image = None
            # apply scaling
            contrast_image = (prev_img * contrast).round().astype(np.uint8)
            # anti-blur



            if amount > 0:
                contrast_image = unsharp_mask(contrast_image, (kernel, kernel), sigma, amount, threshold,
                                              mask_image if args.use_mask else None)
            # apply frame noising
            if args.use_mask or anim_args.use_noise_mask:
                root.noise_mask = compose_mask_with_check(root, args, noise_mask_seq,
                                                               noise_mask_vals,
                                                               Image.fromarray(
                                                                   cv2.cvtColor(contrast_image, cv2.COLOR_BGR2RGB)))
            noised_image = add_noise(contrast_image, noise, int(args.seed), anim_args.noise_type,
                                     (anim_args.perlin_w, anim_args.perlin_h,
                                      anim_args.perlin_octaves,
                                      anim_args.perlin_persistence),
                                     root.noise_mask, args.invert_mask)


            image = Image.fromarray(noised_image)
            print(f"[ Deforum Adding Noise: {noise} ]")

            image = pil2tensor(image)

        return [data, image]


