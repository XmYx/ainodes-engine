import torch

from backend_helpers.torch_helpers.add_noise import add_noise, add_noise_with_mask
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode
from ainodes_frontend.base.settings import handle_ainodes_exception
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend import singleton as gs

OP_NODE_ADD_NOISE = get_next_opcode()
class AddNoiseWidget(QDMNodeContentWidget):
    def initUI(self):

        self.noise_type = self.create_combo_box(['gaussian', 'salt_pepper', 'poisson'], "Noise Type")
        self.noise_amount = self.create_double_spin_box(label_text="Noise Amount", min_val=0.0, max_val=10.0)

        self.create_main_layout(grid=1)

@register_node(OP_NODE_ADD_NOISE)
class AddNoiseNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
    help_text = "Adds Noise to an image"
    op_code = OP_NODE_ADD_NOISE
    op_title = "Add Noise"
    content_label_objname = "add_noise_node"
    category = "base/image"
    NodeContent_class = AddNoiseWidget
    dim = (340, 180)
    output_data_ports = [0]
    exec_port = 1

    custom_input_socket_name = ["IMAGE", "MASK", "EXEC"]

    def __init__(self, scene):
        super().__init__(scene, inputs=[5,5,1], outputs=[5,1])

    def evalImplementation_thread(self, index=0):

        results = []
        tensors = self.getInputData(0)
        mask_tensors = self.getInputData(1)

        noise_type = self.content.noise_type.currentText()
        noise_amount = self.content.noise_amount.value()
        if tensors is not None:
            for tensor in tensors:
                if mask_tensors is not None:
                    result = add_noise_with_mask(tensor, mask_tensors[0], noise_type=noise_type, noise_amount=noise_amount)
                else:
                    result = add_noise(tensor, noise_type=noise_type, noise_amount=noise_amount)
                results.append(result)

        return [torch.stack(results)]

