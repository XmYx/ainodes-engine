from custom_nodes.layouts.image_preview_layout import ImagePreviewLayout
from NodeGraphQt import NodeBaseWidget


class ImagePreviewBaseWidget(NodeBaseWidget):
    def __init__(self, parent=None):
        super(ImagePreviewBaseWidget, self).__init__(parent)

        self.set_name("image_widget")
        self.set_label("Image Widget")
        self.image = ImagePreviewLayout()
        self.set_custom_widget(self.image)
        self.setMinimumHeight(640)
        self.setMinimumWidth(640)
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

