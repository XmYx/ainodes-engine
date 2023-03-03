from NodeGraphQt import NodeBaseWidget
from Qt import QtCore, QtWidgets

from custom_nodes.layouts.image_preview_layout import ImagePreviewLayout


class ImagePreviewBaseWidget(NodeBaseWidget):
    #set_image_signal = QtCore.Signal(object)

    def __init__(self, parent=None):
        super(ImagePreviewBaseWidget, self).__init__(parent)

        self.set_name("image_widget")
        self.set_label("Image Widget")

        # create a container widget to hold the ImagePreviewLayout
        #self.container = QtWidgets.QWidget()
        #layout = QtWidgets.QVBoxLayout(self.container)
        #layout.setContentsMargins(0, 0, 0, 0)
        self.parent = parent
        widget = ImagePreviewLayout()

        self.set_custom_widget(widget)
        self.setMinimumHeight(540)
        self.setMinimumWidth(540)

    def wire_signals(self):
        pass

    def on_btn_go_clicked(self):
        return None
        print('Clicked on node: "{}"'.format(self.node.name()))

    def get_value(self):
        return None

    def set_value(self, text):
        pass
