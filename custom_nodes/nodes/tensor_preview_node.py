import copy

from custom_nodes.auto_base_node import AutoBaseNode
from custom_nodes.base_widgets.tensor_preview_base import TensorPreviewBaseWidget
from NodeGraphQt import BaseNode


class TensorPreviewNode(AutoBaseNode):
    """
    An example of a node with a embedded QLineEdit.
    """

    # unique node identifier.
    __identifier__ = 'nodes.widget'

    # initial default node name.
    NODE_NAME = 'tensor_preview'

    def __init__(self, parent=None):
        super(TensorPreviewNode, self).__init__()

        # create input & output ports
        self.add_input('in_exe')
        self.add_output('out', multi_output=True)
        self.create_property('out', None)
        self.create_property('in_exe', None)

        #self.height = 512
        # create QLineEdit text input widget.
        self.custom = TensorPreviewBaseWidget(self.view)
        self.add_custom_widget(self.custom, tab='Custom')

    def execute(self):
        try:

            image = self.get_property('in_exe')
            print(image)
            img = copy.deepcopy(image)
            #print("YAY, You have found an image", image)

            self.custom.image.set_value(img)
            self.set_property('out', img, push_undo=False)
        except:
            image = None
        super().execute()
    def set_value(self, value):
        #image = copy.deepcopy(value)
        return
        #self.custom.image.set_image_signal.emit(image)
