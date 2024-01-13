import secrets
import numpy as np
import base64
import io

import torch

from ainodes_frontend.base.qimage_ops import pil2tensor, tensor2pil
import openai

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from backend_helpers.torch_helpers.vram_management import offload_to_device
from ainodes_frontend import singleton as gs
VAE_ENCODE = get_next_opcode()

class VAEEncodeWidget(QDMNodeContentWidget):
    def initUI(self):

        self.create_main_layout(grid=2)

@register_node(VAE_ENCODE)
class VAEEncodeNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
    help_text = "XL"
    op_code = VAE_ENCODE
    op_title = "VAE Encode (aiNodes)"
    content_label_objname = "vae_encode_node"
    category = "base/torch"
    NodeContent_class = VAEEncodeWidget
    dim = (340, 340)
    output_data_ports = [0]
    exec_port = 1
    use_gpu = False
    make_dirty = True

    custom_input_socket_name = ["VAE", "IMAGE", "EXEC"]
    def __init__(self, scene):
        super().__init__(scene, inputs=[4,5,1], outputs=[2,1])


    def evalImplementation_thread(self, index=0):
        vae = self.getInputData(0)
        latent = self.getInputData(1)
        latent = latent.movedim(-1, 1)
        tile_x = 512
        tile_y = 512
        overlap = 64

        latent = remove_fourth_channel(latent)

        with torch.inference_mode():
            offload_to_device(vae, gs.device)
            t = vae.encode_tiled_(latent.to(torch.bfloat16), tile_x=tile_x, tile_y=tile_y, overlap=overlap)
        if gs.vram_state in ["low", "medium"]:
            offload_to_device(vae, "cpu")
        #t = vae.encode(latent[:,:,:,:3])


        return [{"samples": t}]


def remove_fourth_channel(tensor):
    """
    Removes the fourth channel from a tensor if it exists.

    Args:
    tensor (torch.Tensor): A tensor of shape [N, C, H, W] where C is the number of channels.

    Returns:
    torch.Tensor: A tensor with the fourth channel removed if it exists.
    """
    # Check if the tensor has at least 4 channels
    if tensor.shape[1] >= 4:
        # Keep all channels except the fourth
        return tensor[:, :3, :, :]
    else:
        # Return the original tensor if it has less than 4 channels
        return tensor