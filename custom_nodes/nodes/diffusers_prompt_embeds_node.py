from NodeGraphQt import BaseNode
from Qt import QtCore

from custom_nodes.auto_base_node import AutoBaseNode
from custom_nodes.base_widgets.prompt_embed_base import PromptEmbedBaseWidget


class PromptEmbedNode(AutoBaseNode):
    """
    An example of a node with a embedded QLineEdit.
    """

    # unique node identifier.
    __identifier__ = 'nodes.widget'

    # initial default node name.
    NODE_NAME = 'exec'
    def __init__(self, parent=None):
        super(PromptEmbedNode, self).__init__()

        # create input & output ports
        self.add_output('prompt_embeds')
        self.create_property('prompt_embeds', None)
        self.add_input('prompt_exe')
        self.create_property('prompt_exe', None)
        self.add_input('pipe')
        self.create_property('pipe', None)
        self.custom = PromptEmbedBaseWidget(self.view, self)
        self.add_custom_widget(self.custom, tab='Custom')

    def execute(self):
        prompt = self.custom.custom.prompt.toPlainText()
        if prompt == "":
            prompt = self.get_property("prompt_exe")
        print(self.get_property('pipe'))
        pipe = self.get_property('pipe')
        prompt_embeds = pipe._encode_prompt(prompt=prompt, device='cuda', num_images_per_prompt=1, do_classifier_free_guidance=True)
        self.set_property('prompt_embeds', prompt_embeds)
        #self.execute_children()
        super().execute()