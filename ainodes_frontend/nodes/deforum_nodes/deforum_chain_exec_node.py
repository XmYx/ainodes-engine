import torch
import os

from backend_helpers.diffusers_helpers.diffusers_helpers import diffusers_models, diffusers_indexed, \
    scheduler_type_values, SchedulerType, get_scheduler_class
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget

OP_NODE_DEFORUM_EXEC_CHAIN = get_next_opcode()

class DeforumExecChainWidget(QDMNodeContentWidget):
    def initUI(self):
        self.create_main_layout(grid=1)

@register_node(OP_NODE_DEFORUM_EXEC_CHAIN)
class DeforumExecChainNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
    help_text = "Deforum Exec Chain"
    op_code = OP_NODE_DEFORUM_EXEC_CHAIN
    op_title = "Deforum - Exec Chain"
    content_label_objname = "deforum_exec_chain_node"
    category = "base/deforum"
    NodeContent_class = DeforumExecChainWidget
    dim = (340, 150)
    output_data_ports = [0]
    mark_dirty = True
    custom_output_socket_name = ["EXEC", "FINAL EXEC"]

    def __init__(self, scene):
        super().__init__(scene, inputs=[6,1], outputs=[1,1])
    def evalImplementation_thread(self, index=0):

        data = self.getInputData(0)
        if data:
            if data.get("frame_idx", 0) < data.get("max_frames", 1):
                return True
        return False

    def onWorkerFinished(self, result, exec=True):
        self.markDirty(False)
        self.busy = False
        if result:
            print("Will execute child 0")
            self.executeChild(0)
        else:
            print("Will execute child 1")

            self.executeChild(1)

