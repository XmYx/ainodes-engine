import copy

from PIL.ImageQt import ImageQt

from custom_nodes.auto_base_node import AutoBaseNode
from custom_nodes.base_widgets.image_preview_base import ImagePreviewBaseWidget
from NodeGraphQt import BaseNode
from Qt import QtCore, QtGui, QtWidgets

from custom_nodes.layouts.image_preview_layout import ImagePreviewLayout


class ImagePreviewNode(AutoBaseNode):
    """
    An example of a node with a embedded QLineEdit.
    """

    # unique node identifier.
    __identifier__ = 'nodes.widget'

    # initial default node name.
    NODE_NAME = 'image_preview'

    def __init__(self, parent=None):
        super(ImagePreviewNode, self).__init__()

        # create input & output ports
        self.add_input('in_exe')
        self.add_output('out', multi_output=True)
        self.create_property('out', None)
        self.create_property('in_exe', None)

        self.custom = ImagePreviewBaseWidget(self.view)
        self.image = self.custom.get_custom_widget().image
        self.add_custom_widget(self.custom, tab='Custom')
        self.pixmap = QtGui.QPixmap()

    def execute(self):
        print(id(self.view), id(self), id(self.custom.get_custom_widget()))
        image = self.get_property('in_exe')
        img = copy.deepcopy(image)
        img2 = copy.deepcopy(image)
        self.set_property('out', img, push_undo=False)
        widget = self.custom.get_custom_widget()
        widget.set_image_signal.emit(img2)
        #self.execute_children()

    @QtCore.Slot(object)
    def set_image(self, image):
        print("image was set")
        image = self.get_property('in_exe')
        image_data_copy = copy.deepcopy(image)
        qimage = ImageQt(image_data_copy)
        pixmap = QtGui.QPixmap().fromImage(qimage)
        widget = self.custom.get_custom_widget()
        widget.image.setPixmap(pixmap)
    def on_input_disconnected(self, in_port, out_port):
        try:
            widget = self.custom.get_custom_widget()
            widget.set_image_signal.disconnect()
            print("dc")
        except:
            print("Already disconnected")
    def on_input_connected(self, in_port, out_port):
        try:
            widget = self.custom.get_custom_widget()

            widget.set_image_signal.connect(self.set_image)
            print("c")
        except:
            print("Already disconnected")