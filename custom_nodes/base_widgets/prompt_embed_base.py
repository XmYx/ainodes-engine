from custom_nodes.layouts.prompt_embed_layout import PromptEmbedsLayout
from NodeGraphQt import NodeBaseWidget

class PromptEmbedBaseWidget(NodeBaseWidget):
    def __init__(self, parent=None, realparent=None):
        super(PromptEmbedBaseWidget, self).__init__(parent)
        self.parent = realparent
        self.set_name("prompt_embed")
        self.set_label("Prompt Embeds")
        self.custom = PromptEmbedsLayout(realparent=self)
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