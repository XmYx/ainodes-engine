
from diffusers import AutoencoderTiny

import torch

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget


OP_NODE_TINYVAE = get_next_opcode()


class TinyVAEWidget(QDMNodeContentWidget):
    def initUI(self):
        # Create a label to display the image
        self.create_main_layout(grid=1)



@register_node(OP_NODE_TINYVAE)
class TinyVAEDecode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/film.png"
    op_code = OP_NODE_TINYVAE
    op_title = "TinyVAE_Decode"
    content_label_objname = "tinyvae_decode_node"
    category = "base/torch"
    make_dirty = True
    NodeContent_class = TinyVAEWidget
    dim = (400, 300)

    custom_input_socket_name = ["VAE", "LATENT", "EXEC"]
    custom_output_socket_name = ["VAE", "IMAGE", "EXEC"]


    def __init__(self, scene):
        super().__init__(scene, inputs=[4,2,1], outputs=[4,5,1])
        self.vae = None
        self.vae_scale_factor = None

    def evalImplementation_thread(self):
        with torch.inference_mode():
            vae = self.getInputData(0)
            latent = self.getInputData(1)

            if not vae:
                self.vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float16).to("cuda")
            else:
                self.vae = vae
            if not self.vae_scale_factor:
                self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            latent = latent["samples"].to("cuda") / self.vae_scale_factor
            decoded = self.vae.decode(latent)
            return [self.vae, (decoded.sample.permute(0, 2, 3, 1) / 2 + 0.5).clamp(0, 1)]
    def remove(self):
        if self.vae:
            self.vae.to('cpu')
            del self.vae