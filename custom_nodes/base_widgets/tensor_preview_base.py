from custom_nodes.layouts.tensor_preview_layout import TensorPreviewLayout
from NodeGraphQt import NodeBaseWidget


class TensorPreviewBaseWidget(NodeBaseWidget):
    def __init__(self, parent=None):
        super(TensorPreviewBaseWidget, self).__init__(parent)

        self.set_name("image_widget")
        self.set_label("Image Widget")
        self.image = TensorPreviewLayout()
        self.set_custom_widget(self.image)
        self.setMinimumHeight(256)
        self.setMinimumWidth(256)
    def wire_signals(self):
        pass
        #widget = self.get_custom_widget()

    def on_btn_go_clicked(self):
        return None
        print('Clicked on node: "{}"'.format(self.node.name()))

    def get_value(self):
        #widget = self.get_custom_widget()
        return None
    def set_value(self, text):
        pass

