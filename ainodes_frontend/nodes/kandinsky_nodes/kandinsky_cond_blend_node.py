from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode
from ainodes_frontend.base.settings import handle_ainodes_exception
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend import singleton as gs

OP_NODE_KANDINSKY_COND_BLEND = get_next_opcode()
class KandinskyCondBlendWidget(QDMNodeContentWidget):
    def initUI(self):

        self.blend = self.create_double_spin_box("Blend", min_val=0.0, max_val=1.0, default_val=1.0)

        self.create_main_layout(grid=1)

@register_node(OP_NODE_KANDINSKY_COND_BLEND)
class KandinskyCondBlendNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
    help_text = "Data objects in aiNodes are simple dictionaries,\n" \
                "that can hold any values under any name.\n" \
                "In most cases, you'll find them drive parameters,\n" \
                "or hold sequences of images. For an example, the\n" \
                "OpenAI node emits it's prompt in a data line,\n" \
                "but you'll find this info in all relevant places."
    op_code = OP_NODE_KANDINSKY_COND_BLEND
    op_title = "Kandinsky Conditioning Blend"
    content_label_objname = "kandinsky_cond_blend_node"
    category = "base/kandinsky"
    NodeContent_class = KandinskyCondBlendWidget
    dim = (340, 180)
    output_data_ports = [0]
    exec_port = 1


    def __init__(self, scene):
        super().__init__(scene, inputs=[3,3,1], outputs=[3,1])

    def evalImplementation_thread(self, index=0):
        conds1 = self.getInputData(0)
        conds2 = self.getInputData(1)
        blend_val = self.content.blend.value()

        return [[addWeighted(conds1[0], conds2[0], blend_val)]]
def addWeighted(tensor1, tensor2, blend_value):
    if blend_value < 0 or blend_value > 1:
        raise ValueError("Blend value should be between 0 and 1.")

    blended_tensor = blend_value * tensor1 + (1 - blend_value) * tensor2
    return blended_tensor