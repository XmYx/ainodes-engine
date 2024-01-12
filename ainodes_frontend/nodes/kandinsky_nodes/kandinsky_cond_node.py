from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode
from ainodes_frontend.base.settings import handle_ainodes_exception
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend import singleton as gs

OP_NODE_KANDINSKY_COND = get_next_opcode()
class KandinskyCondWidget(QDMNodeContentWidget):
    def initUI(self):

        self.prompt = self.create_text_edit("Prompt")
        self.n_prompt = self.create_text_edit("Negative Prompt")
        self.guidance_scale = self.create_double_spin_box("Guidance Scale", min_val=0.1, max_val=25.0, default_val=1.0)

        self.create_main_layout(grid=1)

@register_node(OP_NODE_KANDINSKY_COND)
class KandinskyCondNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
    help_text = "Data objects in aiNodes are simple dictionaries,\n" \
                "that can hold any values under any name.\n" \
                "In most cases, you'll find them drive parameters,\n" \
                "or hold sequences of images. For an example, the\n" \
                "OpenAI node emits it's prompt in a data line,\n" \
                "but you'll find this info in all relevant places."
    op_code = OP_NODE_KANDINSKY_COND
    op_title = "Kandinsky Conditioning"
    content_label_objname = "kandinsky_cond_node"
    category = "base/kandinsky"
    NodeContent_class = KandinskyCondWidget
    dim = (340, 500)
    output_data_ports = [0,1]
    exec_port = 2

    custom_input_socket_name = ["PRIOR", "EXEC"]

    def __init__(self, scene):
        super().__init__(scene, inputs=[4,1], outputs=[3,3,1])
        self.prior = None
        self.decoder = None

    def evalImplementation_thread(self, index=0):

        prior = self.getInputData(0)
        assert prior is not None, "No PRIOR model found"

        prompt = self.content.prompt.toPlainText()
        n_prompt = self.content.n_prompt.toPlainText()
        scale = self.content.guidance_scale.value()
        prior.to("cuda")
        image_embeds, negative_image_embeds = prior(prompt, n_prompt, guidance_scale=scale).to_tuple()
        prior.to("cpu")
        return [[image_embeds], [negative_image_embeds]]
