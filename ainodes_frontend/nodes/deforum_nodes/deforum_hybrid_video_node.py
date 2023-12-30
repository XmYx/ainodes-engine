
# import cv2
# import numpy as np
# from deforum.animation.animation import anim_frame_warp
# from deforum.exttools.depth import DepthModel
# from PIL import Image
# from ai_nodes.ainodes_engine_base_nodes.ainodes_backend import pil2tensor, tensor2pil
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
# from ainodes_frontend import singleton as gs

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
    category = "aiNodes Deforum/DeForum"
    NodeContent_class = DeforumFramewarpWidget
    dim = (240, 120)
    # custom_output_socket_name = ["DATA", "IMAGE", "EXEC"]

    make_dirty = True

    def __init__(self, scene):
        super().__init__(scene, inputs=[6,5,1], outputs=[5,1])

    def evalImplementation_thread(self, index=0):


        data = self.getInputData(0)
        image = self.getInputData(1)

        if data is not None and image is not None:

            pass