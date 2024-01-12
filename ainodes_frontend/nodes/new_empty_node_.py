from qtpy import QtWidgets, QtCore, QtGui
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.base.qimage_ops import pixmap_to_tensor, tensor_image_to_pixmap, tensor2pil, pil2tensor
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from backend_helpers.torch_helpers.torch_gc import torch_gc



"""
When creating a new node, make sure to change the global OP_NODE_UNIQUE to something unique, like OP_NODE_MYVALUE = get_next_opcode(),
then replace in your node class as well. The next steps are renaming your CustomWidget, your CustomNode, then setting the
content_labale_objname to something unique. Feel free to add inputs, outputs, and any custom function inside the evalImplementation_thread().

Use the built in torch_gc function to clear some VRAM, and don't forget to unload your model in the remove function if any.

There are also other methods available: tensor2pil, pil2tensor

Put this file in any folder ending with nodes, and remove the "_" from the end for it to show up on start.
"""
OP_NODE_UNIQUE = get_next_opcode()

from ainodes_frontend import singleton as gs

class CustomWidget(QDMNodeContentWidget):
    def initUI(self):
        """
        Initialize the User Interface for this custom widget.
        This method should be called to set up the layout and widgets inside the node.
        """
        pass

        # Create a label to display the image
        self.create_widgets()
        self.create_main_layout(grid=1)
    def create_widgets(self):
        """
        Create the widgets for this node.
        This method sets up the UI elements like checkboxes, text fields, etc.
        """
        self.create_check_box("Checkbox", spawn="checkbox")



@register_node(OP_NODE_UNIQUE)
class CustomNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/film.png"
    op_code = OP_NODE_UNIQUE
    op_title = "Node Title"
    content_label_objname = "unique_node_id"
    category = "custom/category"
    make_dirty = True
    help_text = "Help text to appear"
    # custom_input_socket_name = ["IMG_0", "EXEC"]
    # custom_output_socket_name = ["IMG_0", "EXEC"]

    def __init__(self, scene):

        """
         Initialize a CustomNode class with a scene, inputs, and outputs.

         Args:
             scene (QGraphicsScene): The scene where the node is displayed.
             inputs (list): A list of input sockets, with values representing their types.
                           1: EXEC
                           2: LATENT
                           3: CONDITIONING
                           4: PIPE/MODEL
                           5: IMAGE
                           6: DATA
                           7: STRING
                           8: INT
                           9: FLOAT
                           Defaults to [2,2].
             outputs (list): A list of output sockets, with values representing their types.
                           1: EXEC
                           2: LATENT
                           3: CONDITIONING
                           4: PIPE/MODEL
                           5: IMAGE
                           6: DATA
                           7: STRING
                           8: INT
                           9: FLOAT
                           Defaults to [1].
         """

        super().__init__(scene, inputs=[5,1], outputs=[5,1])

    def evalImplementation_thread(self):
        """
        The core evaluation logic of the node, to be run in a separate thread.
        This method should implement the processing logic of the node,
        which is executed when the node is evaluated.

        Use self.getInputData(int) to retrieve data from a connected input socket.

        Sockets are numbered from the bottom of the node, starting at 0.
        For example, in a node with 3 inputs, the bottom-most socket is 0, and the top-most is 2.

        When returning from this function the self.onWorkerFinished(result) is called.

        The 'result' argument should always be a list, as this method sets the node's outputs based on this list.
        Each element in the list corresponds to an output socket, in the order of the sockets.
        """

        return [None]
    def remove(self):
        """

        """

        super().remove()
