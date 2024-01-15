
from diffusers import AutoencoderTiny

import torch

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend import singleton as gs

OP_NODE_TINYVAE = get_next_opcode()


class TinyVAEWidget(QDMNodeContentWidget):


    def initUI(self):
        # Create a label to display the image
        self.create_check_box(label_text="XL", accessible_name="is_xl", spawn="is_xl")
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
        self.version = None
    def evalImplementation_thread(self):
        with torch.inference_mode():
            vae = self.getInputData(0)
            latent = self.getInputData(1)
            print(latent["samples"].shape)
            if not vae:
                if not self.vae or self.version != self.content.is_xl.isChecked():
                    vae_version = "madebyollin/taesdxl" if self.content.is_xl.isChecked() else "madebyollin/taesd"
                    print("Loading TinyVAE: ", vae_version)
                    self.vae = AutoencoderTiny.from_pretrained(vae_version, torch_dtype=torch.bfloat16).to("cuda")
                    self.version = self.content.is_xl.isChecked()
                    self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            else:
                self.vae = vae

            if latent is not None:
                latent = latent["samples"].bfloat16().to(gs.device) / self.vae_scale_factor
                decoded = self.vae.decode(latent)
                return [self.vae, (decoded.sample.permute(0, 2, 3, 1) / 2 + 0.5).clamp(0, 1).half()]
            else:
                return [self.vae, None]
    def remove(self):
        if hasattr(self, "vae"):
            if self.vae:
                try:
                    self.vae.to('cpu')
                except:
                    pass
        self.vae = None
        super().remove()