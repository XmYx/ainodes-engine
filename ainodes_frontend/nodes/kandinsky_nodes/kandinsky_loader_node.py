from diffusers import KandinskyV22PriorPipeline, KandinskyV22Pipeline

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode
from ainodes_frontend.base.settings import handle_ainodes_exception
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend import singleton as gs
from backend_helpers.torch_helpers.torch_gc import torch_gc

OP_NODE_KANDINSKY_LOADER = get_next_opcode()
class KandinskyLoaderWidget(QDMNodeContentWidget):
    def initUI(self):
        self.create_main_layout(grid=1)

@register_node(OP_NODE_KANDINSKY_LOADER)
class KandinskyLoaderNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
    help_text = "Data objects in aiNodes are simple dictionaries,\n" \
                "that can hold any values under any name.\n" \
                "In most cases, you'll find them drive parameters,\n" \
                "or hold sequences of images. For an example, the\n" \
                "OpenAI node emits it's prompt in a data line,\n" \
                "but you'll find this info in all relevant places."
    op_code = OP_NODE_KANDINSKY_LOADER
    op_title = "Kandinsky Loader"
    content_label_objname = "kandinsky_loader_node"
    category = "base/kandinsky"
    NodeContent_class = KandinskyLoaderWidget
    dim = (340, 180)
    output_data_ports = [0,1,2]
    exec_port = 3

    custom_output_socket_name = ["PRIOR", "DECODER", "IMG2IMG", "EXEC"]

    def __init__(self, scene):
        super().__init__(scene, inputs=[1], outputs=[4,4,4,1])
        self.prior = None
        self.decoder = None

    def evalImplementation_thread(self, index=0):
        from diffusers import KandinskyV22PriorPipeline, KandinskyV22Pipeline, KandinskyV22Img2ImgPipeline

        if self.prior == None:

            from diffusers import DiffusionPipeline
            import torch

            # self.prior = DiffusionPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-prior",
            #                                                torch_dtype=torch.float16)
            self.prior = KandinskyV22PriorPipeline.from_pretrained(
                "kandinsky-community/kandinsky-2-2-prior", torch_dtype=torch.float16
            )
            # self.decoder = DiffusionPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-decoder",
            #                                              torch_dtype=torch.float16)
            self.decoder = KandinskyV22Pipeline.from_pretrained("kandinsky-community/kandinsky-2-2-decoder",
                                                        torch_dtype=torch.float16)
            self.img2img_decoder = KandinskyV22Img2ImgPipeline(unet = self.decoder.unet,
                                                                scheduler = self.decoder.scheduler,
                                                                movq = self.decoder.movq)

            return [self.prior, self.decoder, self.img2img_decoder]
    def remove(self):
        if self.prior:
            try:
                self.prior.to("cpu")
                self.decoder.to("cpu")
                self.img2img_decoder.to("cpu")
            except:
                pass
            self.prior = None
            self.decoder = None
            self.img2img_decoder = None
        torch_gc()

