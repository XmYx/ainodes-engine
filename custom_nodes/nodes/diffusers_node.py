import copy

from custom_nodes.auto_base_node import AutoBaseNode
from custom_nodes.base_widgets.diffusers_base import DiffusersBaseWidget
from NodeGraphQt import BaseNode
import torch
from diffusers import StableDiffusionPipeline
from Qt import QtCore
import singleton

gs = singleton.Singleton()


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
        self.add_input('in_exe')
        self.create_property('in_exe', None)
        self.add_output("out_image")
        self.create_property("out_image", None)
        self.add_output("out_tensor")
        self.create_property("out_tensor", None)
        self.add_output("clip")
        self.create_property("clip", None)
        self.add_output("unet")
        self.create_property("unet", None)
        self.add_output("vae")
        self.create_property("vae", None)
        self.add_output("pipe_out")
        self.create_property("pipe_out", None)
        # create QLineEdit text input widget.
        self.custom = DiffusersBaseWidget(self.view, self)
        self.add_custom_widget(self.custom, tab='Custom')
        self.custom.custom.set_image_signal.connect(self.set_tensor_output)

    def load_diffusers(self):
        repo_id = "runwayml/stable-diffusion-v1-5"
        gs.obj["pipe"] = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path=repo_id, torch_dtype=torch.float16, safety_checker=None).to("cuda")
        gs.obj["pipe"].enable_xformers_memory_efficient_attention()
        print("Diffusers model loaded")
        #self.set_property("clip", pipe.text_encoder, push_undo=False)
        #self.set_property("unet", pipe.unet, push_undo=False)
        #self.set_property("vae", pipe.vae, push_undo=False)
        self.set_property("pipe_out", "pipe", push_undo=False)
        self.set_children_ports()
    def execute(self,progress_callback=None):
        try:
            #pipe = self.get_property('pipe_out')
            image = gs.obj["pipe"](prompt=self.custom.custom.prompt.toPlainText(), num_inference_steps=self.custom.custom.steps.value(), callback=self.set_tensor_output_signal, callback_steps=1, prompt_embeds=None).images[0]
            #image = Image.open('test.png')
            img = copy.deepcopy(image)
            self.set_property('out_image', img)
            #del pipe
        except:
            pass
        self.execute_children()
        #super().execute()
    @QtCore.Slot(object)
    def set_tensor_output(self, latent):
        returnlatent = latent.cpu()
        #print(returnlatent)
        self.set_property("out_tensor", returnlatent, push_undo=False)
        output_nodes = self.connected_output_nodes()
        for output_port, node_list in output_nodes.items():
            try:
                output_ports = output_port.connected_ports()
                for port in output_ports:
                    name = port.name()
                    own_name = output_port.name()
                    #print("setting output property", own_name, "to input:", name)
                    if "tensor" in own_name:
                        node = port.node()
                        node.set_property(name, returnlatent)
                        if "exe" in name:
                            node.execute()
                        #for node in node_list:
                        #    node.execute()
            except Exception as e:
                print("We have caught an error during processing, your event loop will now stop because:", e)


    def set_tensor_output_signal(self, int1, int2, latent):

        self.custom.custom.set_image_signal.emit(latent)

    def emit_run_signal(self):
        id = self.id
        self._graph.run_node_by_id(id)
