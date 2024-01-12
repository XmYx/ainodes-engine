import torch

from ainodes_frontend.base.qimage_ops import tensor2pil
from ainodes_frontend.nodes.torch_nodes.ksampler_node import get_fixed_seed
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode
from ainodes_frontend.base.settings import handle_ainodes_exception
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend import singleton as gs

OP_NODE_KANDINSKY_COND_INTERP = get_next_opcode()
OP_NODE_KAND_COND_INTERP_VANILLA_PROMPT = get_next_opcode()
OP_NODE_KAND_COND_INTERP_VANILLA_IMAGE = get_next_opcode()
OP_NODE_KAND_COND_INTERP_VANILLA_ASSEMBLER = get_next_opcode()
OP_NODE_KAND_COND_INTERP_VANILLA_COND = get_next_opcode()
class KandinskyCondInterpWidget(QDMNodeContentWidget):
    def initUI(self):
        self.blend = self.create_spin_box("Blend", min_val=0, max_val=4096, default_val=15)
        self.exp = self.create_check_box("Exponential")
        self.create_main_layout(grid=1)

class KandinskyCondPromptWidget(QDMNodeContentWidget):
    def initUI(self):
        self.prompt = self.create_text_edit("Prompt")
        self.weight = self.create_double_spin_box("Weight", min_val=0, max_val=1.0, default_val=0.5)
        self.create_main_layout(grid=1)
class KandinskyCondImageWidget(QDMNodeContentWidget):
    def initUI(self):
        self.weight = self.create_double_spin_box("Weight", min_val=0, max_val=1.0, default_val=0.5)
        self.create_main_layout(grid=1)
class KandinskyCondVanillaInterpWidget(QDMNodeContentWidget):
    def initUI(self):
        self.seed = self.create_line_edit("Seed:", placeholder="Leave empty for random seed")

        self.neg_prior_prompt = self.create_text_edit("Prompt")
        self.neg_prompt = self.create_text_edit("Negative Prompt")
        self.guidance_scale = self.create_double_spin_box("Guidance Scale", min_val=0.1, max_val=25.0, default_val=4.0)
        self.steps = self.create_spin_box("Steps", min_val=1, max_val=250000, default_val=25)

        self.create_main_layout(grid=1)

@register_node(OP_NODE_KANDINSKY_COND_INTERP)
class KandinskyCondBlendNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
    help_text = "Data objects in aiNodes are simple dictionaries"
    op_code = OP_NODE_KANDINSKY_COND_INTERP
    op_title = "Kandinsky Conditioning Interpolation"
    content_label_objname = "kandinsky_cond_interp_node"
    category = "base/kandinsky"
    NodeContent_class = KandinskyCondInterpWidget
    dim = (340, 180)
    output_data_ports = [0]
    exec_port = 1


    def __init__(self, scene):
        super().__init__(scene, inputs=[3,3,1], outputs=[3,1])

    def evalImplementation_thread(self, index=0):
        conds1 = self.getInputData(0)
        conds2 = self.getInputData(1)
        divisions = self.content.blend.value()
        exp = self.content.exp.isChecked()

        return [calculate_blended_conditionings(conds1[0], conds2[0], divisions, exp)]
@register_node(OP_NODE_KAND_COND_INTERP_VANILLA_PROMPT)
class KandinskyCondPromptNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
    help_text = "Data objects in aiNodes are simple dictionaries"
    op_code = OP_NODE_KAND_COND_INTERP_VANILLA_PROMPT
    op_title = "Kandinsky Conditioning Prompt"
    content_label_objname = "kandinsky_cond_prompt_node"
    category = "base/kandinsky"
    NodeContent_class = KandinskyCondPromptWidget
    dim = (340, 220)
    output_data_ports = [0]
    exec_port = 1


    def __init__(self, scene):
        super().__init__(scene, inputs=[6,1], outputs=[6,1])

    def evalImplementation_thread(self, index=0):
        prompt = self.content.prompt.toPlainText()
        weight = self.content.weight.value()
        data = self.getInputData(0)
        data_chunk = {"prompt":prompt,
                      "weight":weight}
        if data:
            if "cond_interp" in data:
                data["cond_interp"].append(data_chunk)
            else:
                data["cond_interp"] = [data_chunk]
        else:
            data = {"cond_interp":[data_chunk]}
        return [data]
@register_node(OP_NODE_KAND_COND_INTERP_VANILLA_IMAGE)
class KandinskyCondImageNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
    help_text = "Data objects in aiNodes are simple dictionaries"
    op_code = OP_NODE_KAND_COND_INTERP_VANILLA_IMAGE
    op_title = "Kandinsky Conditioning Image"
    content_label_objname = "kandinsky_cond_image_node"
    category = "base/kandinsky"
    NodeContent_class = KandinskyCondImageWidget
    dim = (340, 180)
    output_data_ports = [0]
    exec_port = 1

    def __init__(self, scene):
        super().__init__(scene, inputs=[6,5,1], outputs=[6,1])

    def evalImplementation_thread(self, index=0):
        image = self.getInputData(1)
        image = tensor2pil(image[0])
        weight = self.content.weight.value()
        data = self.getInputData(0)
        data_chunk = {"image":image,
                      "weight":weight}
        if data:
            if "cond_interp" in data:
                data["cond_interp"].append(data_chunk)
            else:
                data["cond_interp"] = [data_chunk]
        else:
            data = {"cond_interp":[data_chunk]}
        return [data]
@register_node(OP_NODE_KAND_COND_INTERP_VANILLA_COND)
class KandinskyCondVanillaInterpNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
    help_text = "Data objects in aiNodes are simple dictionaries"
    op_code = OP_NODE_KAND_COND_INTERP_VANILLA_COND
    op_title = "Kandinsky Conditioning Interpolation (Vanilla)"
    content_label_objname = "kandinsky_cond_vanilla_interp_node"
    category = "base/kandinsky"
    NodeContent_class = KandinskyCondVanillaInterpWidget
    dim = (340, 460)
    output_data_ports = [0,1]
    exec_port = 2
    custom_input_socket_name = ["DATA", "PRIOR", "EXEC"]

    def __init__(self, scene):
        super().__init__(scene, inputs=[6,4,1], outputs=[3,3,1])

    def evalImplementation_thread(self, index=0):
        data = self.getInputData(0)
        prior = self.getInputData(1)
        images_texts = []
        weights = []
        if "cond_interp" in data:
            for data_chunk in data["cond_interp"]:
                if "image" in data_chunk:
                    images_texts.append(data_chunk["image"])
                else:
                    images_texts.append(data_chunk["prompt"])
                weights.append(data_chunk["weight"])
        prior.to(gs.device.type)

        self.seed = self.content.seed.text()
        try:
            self.seed = int(self.seed)
        except:
            self.seed = get_fixed_seed('')

        args = {"num_inference_steps": self.content.steps.value(),
                "generator":  torch.Generator(gs.device.type).manual_seed(self.seed),
                #"latents":  None,
                "negative_prior_prompt":  self.content.neg_prior_prompt.toPlainText(),
                "negative_prompt":  self.content.neg_prompt.toPlainText(),
                "guidance_scale":  self.content.guidance_scale.value()}
        
        
        image_embeds, negative_image_embeds = prior.interpolate(images_texts, weights, **args).to_tuple()
        prior.to("cpu")
        return [[image_embeds], [negative_image_embeds]]



def calculate_blended_conditionings(conditioning_to, conditioning_from, divisions, exp=False):

    if len(conditioning_from) > 1:
        print(
            "Warning: ConditioningAverage conditioning_from contains more than 1 cond, only the first one will actually be applied to conditioning_to.")

    alpha_values = torch.linspace(0, 1, divisions + 2)#  [1:-1]  # Exclude 0 and 1
    #print(alpha_values)

    if exp:
        alpha_values = (torch.exp(alpha_values) - 1) / 2
        #print(alpha_values)


    blended_conditionings = []
    for alpha in alpha_values:
        n = addWeighted(conditioning_to, conditioning_from, alpha)
        blended_conditionings.append(n)


    return blended_conditionings
def addWeighted(tensor1, tensor2, blend_value):
    if blend_value < 0 or blend_value > 1:
        raise ValueError("Blend value should be between 0 and 1.")

    blended_tensor = blend_value * tensor1 + (1 - blend_value) * tensor2
    return blended_tensor