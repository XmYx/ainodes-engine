
import cv2
import numpy as np

from deforum.models import DepthModel
from deforum.utils.deforum_framewarp_utils import anim_frame_warp

# from deforum.animation.animation import anim_frame_warp
# from deforum.exttools.depth import DepthModel

from PIL import Image
from ainodes_frontend.base.qimage_ops import pil2tensor, tensor2pil
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend import singleton as gs

OP_NODE_DEFORUM_FRAMEWARP = get_next_opcode()


class DeforumFramewarpWidget(QDMNodeContentWidget):
    def initUI(self):
        self.create_main_layout(grid=1)

@register_node(OP_NODE_DEFORUM_FRAMEWARP)
class DeforumFramewarpNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
    help_text = "Deforum Frame Warp"
    op_code = OP_NODE_DEFORUM_FRAMEWARP
    op_title = "Deforum Frame Warp"
    content_label_objname = "deforum_framewarp_node"
    category = "Deforum"
    NodeContent_class = DeforumFramewarpWidget
    dim = (240, 120)
    custom_output_socket_name = ["DATA", "IMAGE", "MASK", "EXEC"]

    make_dirty = True

    def __init__(self, scene):
        super().__init__(scene, inputs=[6,5,1], outputs=[6,5,5,1])
        self.depth_model = None
        self.algo = ""
    def evalImplementation_thread(self, index=0):
        np_image = None
        data = self.getInputData(0)
        image = self.getInputData(1)
        if image is not None:
            if image.shape[0] > 1:
                for img in image:
                    np_image = tensor2np(img)
            else:
                np_image = tensor2np(image)

            np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)


            args = data.get("args")
            anim_args = data.get("anim_args")
            keys = data.get("keys")
            frame_idx = data.get("frame_idx")

            predict_depths = (
                                         anim_args.animation_mode == '3D' and anim_args.use_depth_warping) or anim_args.save_depth_maps
            predict_depths = predict_depths or (
                    anim_args.hybrid_composite and anim_args.hybrid_comp_mask_type in ['Depth', 'Video Depth'])
            if self.depth_model == None or self.algo != anim_args.depth_algorithm:
                self.algo = anim_args.depth_algorithm
                if predict_depths:
                    keep_in_vram = True
                    # device = ('cpu' if cmd_opts.lowvram or cmd_opts.medvram else self.root.device)
                    # TODO Set device in root in webui
                    device = gs.device.type
                    self.depth_model = DepthModel("models/other", device, True,
                                             keep_in_vram=keep_in_vram,
                                             depth_algorithm=anim_args.depth_algorithm, Width=args.W,
                                             Height=args.H,
                                             midas_weight=anim_args.midas_weight)

                    # depth-based hybrid composite mask requires saved depth maps
                    if anim_args.hybrid_composite != 'None' and anim_args.hybrid_comp_mask_type == 'Depth':
                        anim_args.save_depth_maps = True
                else:
                    self.depth_model = None
                    anim_args.save_depth_maps = False

            if self.depth_model != None and not predict_depths:
                self.depth_model = None



            warped_np_img, depth, mask = anim_frame_warp(np_image, args, anim_args, keys, frame_idx, depth_model=self.depth_model, depth=None, device='cuda',
                            half_precision=True)

            image = Image.fromarray(cv2.cvtColor(warped_np_img, cv2.COLOR_BGR2RGB))

            tensor = pil2tensor(image)

            if mask is not None:
                mask = mask.cpu()
                #mask = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)

            # print(mask.shape)
            #
            # print(mask)
            #
            mask = mask.mean(dim=0, keepdim=False)
            mask[mask > 1e-05] = 1
            mask[mask < 1e-05] = 0

            # print(mask)
            # print(mask.shape)


            # from ai_nodes.ainodes_engine_base_nodes.ainodes_backend.resizeRight import resizeright
            # from ai_nodes.ainodes_engine_base_nodes.ainodes_backend.resizeRight import interp_methods
            # mask = resizeright.resize(mask, scale_factors=None,
            #                                     out_shape=[mask.shape[0], int(mask.shape[1] // 8), int(mask.shape[2] // 8)
            #                                             ],
            #                                     interp_method=interp_methods.lanczos3, support_sz=None,
            #                                     antialiasing=True, by_convs=True, scale_tolerance=None,
            #                                     max_numerator=10, pad_mode='reflect')
            # print(mask.shape)
            return [data, tensor, mask[0].unsqueeze(0)]
        else:
            return [data, image, None]


def tensor2np(img):

    np_img = np.array(tensor2pil(img))

    return np_img

