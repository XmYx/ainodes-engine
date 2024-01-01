import cv2
import numpy as np
from PIL import Image

from deforum.utils.image_utils import maintain_colors
from ainodes_frontend.base.qimage_ops import pil2tensor, tensor2pil
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget

OP_NODE_DEFORUM_COLORMATCH = get_next_opcode()

class DeforumColorMatchWidget(QDMNodeContentWidget):
    def initUI(self):
        self.create_main_layout(grid=1)


@register_node(OP_NODE_DEFORUM_COLORMATCH)
class DeforumColorMatchNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/conditioning.png"
    op_code = OP_NODE_DEFORUM_COLORMATCH
    op_title = "Deforum ColorMatch Node"
    content_label_objname = "deforum_colormatch_node"
    category = "base/deforum"
    NodeContent_class = DeforumColorMatchWidget
    dim = (240, 140)

    make_dirty = True
    custom_input_socket_name = ["DATA", "IMAGE", "EXEC"]
    custom_output_socket_name = ["DATA", "IMAGE", "EXEC"]


    def __init__(self, scene):
        super().__init__(scene, inputs=[6,5,1], outputs=[6,5,1])
        self.color_match_sample = None
    #@QtCore.Slot()
    def evalImplementation_thread(self, index=0, prompt_override=None):

        data = self.getInputData(0)
        image = self.getInputData(1)

        if image is not None:
            anim_args = data.get("anim_args")
            image = np.array(tensor2pil(image))
            #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


            if anim_args.color_coherence != 'None' and self.color_match_sample is not None:
                image = maintain_colors(image, self.color_match_sample, anim_args.color_coherence)
            print(f"[ Deforum Color Coherence: {anim_args.color_coherence} ]")
            if self.color_match_sample is None:
                self.color_match_sample = image.copy()
            if anim_args.color_force_grayscale:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = pil2tensor(Image.fromarray(image))
            return [data, image]
        else:
            return [data, image]

    def clearOutputs(self):
        self.color_match_sample = None
        super().clearOutputs()




            # keys = data.get("keys")
            # args = data.get("args")
            # anim_args = data.get("anim_args")
            # root = data.get("root")
            # frame_idx = data.get("frame_idx")
            # noise = keys.noise_schedule_series[frame_idx]
            # kernel = int(keys.kernel_schedule_series[frame_idx])
            # sigma = keys.sigma_schedule_series[frame_idx]
            # amount = keys.amount_schedule_series[frame_idx]
            # threshold = keys.threshold_schedule_series[frame_idx]
            # contrast = keys.contrast_schedule_series[frame_idx]
            # if anim_args.use_noise_mask and keys.noise_mask_schedule_series[frame_idx] is not None:
            #     noise_mask_seq = keys.noise_mask_schedule_series[frame_idx]
            # else:
            #     noise_mask_seq = None
            # mask_vals = {}
            # noise_mask_vals = {}
            #
            # mask_vals['everywhere'] = Image.new('1', (args.W, args.H), 1)
            # noise_mask_vals['everywhere'] = Image.new('1', (args.W, args.H), 1)
            #
            # from ai_nodes.ainodes_engine_deforum_nodes.deforum_nodes.deforum_framewarp_node import tensor2np
            # prev_img = tensor2np(image)
            # mask_image = None
            # # apply scaling
            # contrast_image = (prev_img * contrast).round().astype(np.uint8)
            # # anti-blur
            #
            #
            #
            # if amount > 0:
            #     contrast_image = unsharp_mask(contrast_image, (kernel, kernel), sigma, amount, threshold,
            #                                   mask_image if args.use_mask else None)
            # # apply frame noising
            # if args.use_mask or anim_args.use_noise_mask:
            #     root.noise_mask = compose_mask_with_check(root, args, noise_mask_seq,
            #                                                    noise_mask_vals,
            #                                                    Image.fromarray(
            #                                                        cv2.cvtColor(contrast_image, cv2.COLOR_BGR2RGB)))
            # noised_image = add_noise(contrast_image, noise, int(args.seed), anim_args.noise_type,
            #                          (anim_args.perlin_w, anim_args.perlin_h,
            #                           anim_args.perlin_octaves,
            #                           anim_args.perlin_persistence),
            #                          root.noise_mask, args.invert_mask)
            #
            #
            # image = Image.fromarray(noised_image)
            #
            # tensor = pil2tensor(image)
