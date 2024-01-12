import torch

from ainodes_frontend.base.qimage_ops import pil2tensor, tensor2pil
from ainodes_frontend.nodes.torch_nodes.ksampler_node import get_fixed_seed

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode
from ainodes_frontend.base.settings import handle_ainodes_exception
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend import singleton as gs

OP_NODE_KANDINSKY_SAMPLER = get_next_opcode()
class KandinskySamplerWidget(QDMNodeContentWidget):
    def initUI(self):
        #self.guidance_scale = self.create_double_spin_box(min_val=0.1, max_val=25.0, default_val=1.0)
        self.seed = self.create_line_edit("Seed:", placeholder="Leave empty for random seed")
        self.steps = self.create_spin_box("Steps:", 1, 10000, 25)
        self.cfg_scale = self.create_double_spin_box("Guidance Scale:", min_val=0.0, max_val=1000.0, default_val=4.0)
        self.strength = self.create_double_spin_box("Strength:", min_val=0.0, max_val=1.0, default_val=0.85)
        self.w_param = self.create_spin_box("Width:", 64, 2048, 512, 64)
        self.h_param = self.create_spin_box("Height:", 64, 2048, 512, 64)
        self.use_feedback = self.create_check_box("Image Feedback")
        self.create_main_layout(grid=1)

@register_node(OP_NODE_KANDINSKY_SAMPLER)
class KandinskySamplerNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
    help_text = "Data objects in aiNodes are simple dictionaries,\n" \
                "that can hold any values under any name.\n" \
                "In most cases, you'll find them drive parameters,\n" \
                "or hold sequences of images. For an example, the\n" \
                "OpenAI node emits it's prompt in a data line,\n" \
                "but you'll find this info in all relevant places."
    op_code = OP_NODE_KANDINSKY_SAMPLER
    op_title = "Kandinsky Sampler"
    content_label_objname = "kandinsky_sampler_node"
    category = "base/kandinsky"
    NodeContent_class = KandinskySamplerWidget
    dim = (340, 600)
    output_data_ports = [0]
    exec_port = 1

    custom_input_socket_name = ["DECODER", "COND", "COND", "IMG2IMG", "IMAGE", "EXEC"]

    def __init__(self, scene):
        super().__init__(scene, inputs=[4,3,3,4,5,1], outputs=[5,1])

    def evalImplementation_thread(self, index=0):
        decoder = self.getInputData(0)
        image_embeds = self.getInputData(1)
        negative_image_embeds = self.getInputData(2)
        images = []
        input_image = self.getInputData(4)
        if input_image is not None:
            decoder = self.getInputData(3)
        assert decoder is not None, "No DECODER model found"
        assert len(image_embeds) == len(negative_image_embeds), "Make sure to pass the same amount of positive and negative conditionings"
        total_imgs = len(image_embeds)
        index = 0
        use_feedback = self.content.use_feedback.isChecked()
        for image_embed, negative_image_embed in zip(image_embeds, negative_image_embeds):
            num_steps = self.content.steps.value()
            guidance_scale = self.content.cfg_scale.value()
            h = self.content.h_param.value()
            w = self.content.w_param.value()
            strength = self.content.strength.value()
            self.seed = self.content.seed.text()
            try:
                self.seed = int(self.seed)
            except:
                self.seed = get_fixed_seed('')
            generator = torch.Generator(gs.device.type).manual_seed(self.seed)

            args = {"image_embeds":image_embed,
                    "negative_image_embeds":negative_image_embed,
                    "height":h,
                    "width":w,
                    "generator":generator,
                    "num_inference_steps":num_steps,
                    "guidance_scale":guidance_scale,
                    }

            #scale = self.content.guidance_scale.value()
            if index == 0:
                decoder.to("cuda")
            if input_image is not None:
                args["image"] = tensor2pil(input_image[0])
            if use_feedback and len(images) > 0:
                decoder = self.getInputData(3)
                args["image"] = tensor2pil(images[len(images) - 1])
                args["strength"] = strength
            image = decoder(**args).images[0]
            if index == total_imgs:
                decoder.to("cpu")
            images.append(pil2tensor(image))
            print(images[index].shape)
            index += 1
        output = torch.stack(images, dim=0)
        print(output.shape)
        return [output]
