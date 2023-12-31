#from qtpy.QtWidgets import QLineEdit, QLabel, QPushButton, QFileDialog, QVBoxLayout
from ainodes_frontend.node_engine.node_graphics_node import QDMGraphicsBGNode, QDMGraphicsBGInfoNode
from qtpy import QtWidgets, QtGui

from ainodes_frontend.base import register_node, get_next_opcode, AiDummyNode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend.node_engine.utils import dumpException

OP_NODE_BG = get_next_opcode()

@register_node(OP_NODE_BG)
class BGNode(AiDummyNode):
    icon = "ainodes_frontend/icons/base_nodes/bg.png"
    op_code = OP_NODE_BG
    op_title = "Bg Node"
    help_text = "Simple cosmetic background object\n" \
                "that drags the nodes placed on it."
    content_label_objname = "bg_node"
    category = "aiNodes Base/Background"

    def __init__(self, scene):
        super().__init__(scene, inputs=[], outputs=[])
        pass
        # Create a worker object
    def initInnerClasses(self):
        self.grNode = QDMGraphicsBGNode(self)
        self.grNode.setZValue(-2)

    def evalImplementation(self, index=0):
        return None

    def onMarkedDirty(self):
        self.value = None

    def serialize(self):
        res = super().serialize()
        res['color'] = self.grNode._brush_background.color().name(QtGui.QColor.NameFormat.HexArgb)
        res['width'] = self.grNode.width
        res['height'] = self.grNode.height
        return res

    def deserialize(self, data, hashmap={}, restore_id=False):
        res = super().deserialize(data, hashmap)
        try:
            deserialized_color = QtGui.QColor(data['color'])
            deserialized_brush = QtGui.QBrush(deserialized_color)
            self.grNode._brush_background = deserialized_brush
            self.grNode.width = (data['width'])
            self.grNode.height = (data['height'])
            self.grNode._sizer.set_pos(self.grNode.width, self.grNode.height)
            return True & res
        except Exception as e:
            dumpException(e)
        return res

class InfoWidget(QDMNodeContentWidget):
    def initUI(self):
        self.create_widgets()
        self.create_main_layout(grid=1)
    def create_widgets(self):
        self.text = self.create_text_edit("Info")
    def resize(self, size):
        self.setMinimumHeight(int(size["height"]) - 24)
        self.setMaximumHeight(int(size["height"]) - 24)
        self.setMinimumWidth(int(size["width"]) - 24)
        self.setMaximumWidth(int(size["width"]) - 24)

OP_NODE_BG_INFO = get_next_opcode()

@register_node(OP_NODE_BG_INFO)
class BGINFONode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/info.png"
    op_code = OP_NODE_BG_INFO
    op_title = "Bg Node Info"
    content_label_objname = "bg_node_info"
    category = "aiNodes Base/Background"

    def __init__(self, scene):
        super().__init__(scene, inputs=[], outputs=[])
        pass
        # Create a worker object
    def initInnerClasses(self):
        self.content = InfoWidget(self)
        self.grNode = QDMGraphicsBGInfoNode(self)
        self.grNode.setZValue(-2)




    def evalImplementation(self, index=0):
        return None

    def onMarkedDirty(self):
        self.value = None

    def serialize(self):
        res = super().serialize()
        res['color'] = self.grNode._brush_background.color().name(QtGui.QColor.NameFormat.HexArgb)
        res['width'] = self.grNode.width
        res['height'] = self.grNode.height
        return res

    def deserialize(self, data, hashmap={}, restore_id=False):
        res = super().deserialize(data, hashmap)
        try:
            deserialized_color = QtGui.QColor(data['color'])
            deserialized_brush = QtGui.QBrush(deserialized_color)
            self.grNode._brush_background = deserialized_brush
            self.grNode.width = (data['width'])
            self.grNode.height = (data['height'])
            self.grNode._sizer.set_pos(self.grNode.width, self.grNode.height)

            size = {"width": self.grNode._width,
                    "height": self.grNode._height}
            self.content.resize(size)

            return True & res
        except Exception as e:
            dumpException(e)
        return res








