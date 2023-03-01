from custom_nodes.layouts.diffusers_layout import DiffusersLayout
from NodeGraphQt import NodeBaseWidget

class DiffusersBaseWidget(NodeBaseWidget):
    def __init__(self, parent=None, realparent=None):
        super(DiffusersBaseWidget, self).__init__(parent)
        self.parent = realparent
        self.set_name("diffusers_widget")
        self.set_label("Diffusers Widget")
        self.custom = DiffusersLayout(realparent=self)
        self.set_custom_widget(self.custom)
        self.setMinimumHeight(320)
        self.setMinimumWidth(320)
        self.parent.pipe = None

    def wire_signals(self):
        pass

    def on_btn_go_clicked(self):
        print('Clicked on node: "{}"'.format(self.node.name()))

    def get_value(self):
        return self.parent.pipe
    def set_value(self, text):
        return True