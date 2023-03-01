import copy

from custom_nodes.auto_base_node import AutoBaseNode
from custom_nodes.base_widgets.diffusers_base import DiffusersBaseWidget
from NodeGraphQt import BaseNode
import torch
from diffusers import StableDiffusionPipeline


class DiffusersNode(AutoBaseNode):
    """
    An example of a node with a embedded QLineEdit.
    """

    # unique node identifier.
    __identifier__ = 'nodes.widget'

    # initial default node name.
    NODE_NAME = 'image'

    def __init__(self):
        super(DiffusersNode, self).__init__()

        # create input & output ports
        self.add_input('in')
        self.create_property('in', None)
        self.add_output("out")
        self.create_property("out", None)
        # create QLineEdit text input widget.
        self.custom = DiffusersBaseWidget(self.view, self)
        self.add_custom_widget(self.custom, tab='Custom')
    def load_diffusers(self):
        repo_id = "runwayml/stable-diffusion-v1-5"
        self.pipe = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path=repo_id, torch_dtype=torch.float16, safety_checker=None).to("cuda")
        self.pipe.enable_xformers_memory_efficient_attention()
        print("Diffusers model loaded")
    def execute(self):
        try:
            image = self.pipe(prompt=self.custom.custom.prompt.toPlainText(), num_inference_steps=self.custom.custom.steps.value()).images[0]
            #image = Image.open('test.png')
            img = copy.deepcopy(image)
            self.set_property('out', img)
        except:
            pass
        self.execute_children()
    def emit_run_signal(self):
        self._graph.startsignal.emit()
