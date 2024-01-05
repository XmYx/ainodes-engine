import copy

import cv2
import numpy as np

# import cv2
# import numpy as np
# from deforum.animation.animation import anim_frame_warp
# from deforum.exttools.depth import DepthModel
# from PIL import Image
# from ai_nodes.ainodes_engine_base_nodes.ainodes_backend import pil2tensor, tensor2pil
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode
from ainodes_frontend.base.qimage_ops import tensor2pil, pil2tensor
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from deforum.generators.deforum_flow_generator import get_flow_from_images
from deforum.models import RAFT
from deforum.utils.image_utils import image_transform_optical_flow

# from ainodes_frontend import singleton as gs

OP_NODE_DEFORUM_HYBRID = get_next_opcode()


class DeforumHybridWidget(QDMNodeContentWidget):
    def initUI(self):
        self.create_main_layout(grid=1)

@register_node(OP_NODE_DEFORUM_HYBRID)
class DeforumHybridNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
    help_text = "Deforum Hybrid Node"
    op_code = OP_NODE_DEFORUM_HYBRID
    op_title = "Deforum Hybrid Node"
    content_label_objname = "deforum_hybrid_node"
    category = "base/deforum"
    NodeContent_class = DeforumHybridWidget
    dim = (240, 120)
    custom_input_socket_name = ["DATA", "IMAGE", "REF IMAGE", "EXEC"]

    make_dirty = True

    def __init__(self, scene):
        super().__init__(scene, inputs=[6,5,5,1], outputs=[5,1])
        self.prev_image = None
        self.flow = None
        self.raft_model = RAFT()

    def evalImplementation_thread(self, index=0):


        data = self.getInputData(0)
        image = self.getInputData(1)
        image_2 = self.getInputData(1)

        flow_factor = data["keys"].hybrid_flow_factor_schedule_series[data["frame_index"]]

        pil_image = np.array(tensor2pil(image)).astype(np.uint8)
        bgr_image = cv2.cvtColor(pil_image, cv2.COLOR_RGB2BGR)

        methods = ['RAFT', 'DIS Medium', 'DIS Fine', 'Farneback']
        if image_2 is None:
            if self.prev_image is None:
                self.prev_image = bgr_image
                return [image]
            else:
                self.flow = get_flow_from_images(self.prev_image, bgr_image, methods[0], self.raft_model, self.flow)

                self.prev_image = copy.deepcopy(bgr_image)

                bgr_image = image_transform_optical_flow(bgr_image, self.flow, flow_factor)

                rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

                return [pil2tensor(rgb_image)]
        else:
            pil_image_ref = np.array(tensor2pil(image_2)).astype(np.uint8)
            bgr_image_ref = cv2.cvtColor(pil_image_ref, cv2.COLOR_RGB2BGR)
            if self.prev_image is None:
                self.prev_image = bgr_image_ref
                return [image]
            else:
                self.flow = get_flow_from_images(self.prev_image, bgr_image_ref, methods[0], self.raft_model, self.flow)
                bgr_image = image_transform_optical_flow(bgr_image, self.flow, flow_factor)
                rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
                return [pil2tensor(rgb_image)]

