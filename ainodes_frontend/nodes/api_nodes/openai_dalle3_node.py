import secrets
import subprocess
from typing import Literal

import numpy as np
import torch
import base64
import io
from ainodes_frontend.base.qimage_ops import pil2tensor, tensor2pil
import openai

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget

OP_NODE_OPENAI_DALLE3 = get_next_opcode()


class OpenAiDalle3Widget(QDMNodeContentWidget):
    def initUI(self):
        self.prompt = self.create_text_edit("Prompt", placeholder="Prompt")
        self.seed = self.create_line_edit("Seed")
        self.resolution = self.create_combo_box(["1024x1024", "1024x1792", "1792x1024"], "Resolution")
        self.create_main_layout(grid=2)

@register_node(OP_NODE_OPENAI_DALLE3)
class OpenAIDalle3Node(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
    help_text = "XL"
    op_code = OP_NODE_OPENAI_DALLE3
    op_title = "OpenAI Dalle3"
    content_label_objname = "openai_dalle3_node"
    category = "base/api"
    NodeContent_class = OpenAiDalle3Widget
    dim = (340, 340)
    output_data_ports = [0,1]
    exec_port = 1
    use_gpu = False
    make_dirty = True
    def __init__(self, scene):
        super().__init__(scene, inputs=[1], outputs=[5,7,1])
        self.client = openai.OpenAI()
        self.seed = 0
        self.image = None
        self.revised_prompt = None

    def evalImplementation_thread(self, index=0):
        from PIL import Image
        from torchvision.transforms import functional as TF
        data = self.getAllInputs()

        #print(data)

        prompt = data.get("Prompt")
        seed = data.get("Seed")
        resolution = data.get("Resolution")

        try:
            seed = int(seed)
        except:
            seed = -1

        if seed == -1:
            seed = secrets.randbelow(np.iinfo(np.int32).max)
        if self.isDirty() or self.seed != seed:
            self.seed = seed
            r0 = self.client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size=resolution,
                quality="hd",  # "standard"
                n=1,
                response_format="b64_json"
            )
            im0 = Image.open(io.BytesIO(base64.b64decode(r0.data[0].b64_json)))

            im1 = pil2tensor(im0.convert("RGBA"))

            #im1 = TF.to_tensor(im0.convert("RGBA"))
            #im1[:3, im1[3, :, :] == 0] = 0
            self.revised_prompt = r0.data[0].revised_prompt

            self.image = im1

        return [self.image, self.revised_prompt]

    def remove(self):
        super().remove()
