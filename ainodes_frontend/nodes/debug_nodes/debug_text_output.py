from qtpy.QtWidgets import QLineEdit
from qtpy.QtCore import Qt
from ainodes_frontend.nodes.base.node_config import register_node, OP_NODE_DEBUG_OUTPUT
from ainodes_frontend.nodes.base.ai_node_base import CalcNode, CalcGraphicsNode
from ainodes_backend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_backend.node_engine.utils import dumpException
from qtpy import QtWidgets


class TextOutputWidget(QDMNodeContentWidget):
    def initUI(self):
        self.edit = QLineEdit("1", self)
        self.edit.setAlignment(Qt.AlignHCenter)
        layout = QtWidgets.QVBoxLayout(self)

        layout.setContentsMargins(15, 0, 15, 0)
        layout.addWidget(self.edit)
        self.setLayout(layout)
        #self.edit.setObjectName(self.node.content_label_objname)
    def serialize(self):
        res = super().serialize()
        #res['value'] = self.edit.text()
        return res

    def deserialize(self, data, hashmap={}):
        res = super().deserialize(data, hashmap)
        try:
            value = data['value']
            #self.image.setPixmap(value)
            return True & res
        except Exception as e:
            dumpException(e)
        return res


@register_node(OP_NODE_DEBUG_OUTPUT)
class DebugOutputNode(CalcNode):
    icon = "icons/out.png"
    op_code = OP_NODE_DEBUG_OUTPUT
    op_title = "Output"
    content_label_objname = "debug_output_node"
    category = "debug"


    def __init__(self, scene):
        super().__init__(scene, inputs=[1], outputs=[])
        self.eval()

    def initInnerClasses(self):
        self.content = TextOutputWidget(self)
        self.content.setContentsMargins(5,5,5,5)
        self.grNode = CalcGraphicsNode(self)


    def evalImplementation(self, index=0):
        input_node, other_index = self.getInput(0)
        #print(other_index)
        if not input_node:
            self.grNode.setToolTip("Input is not connected")
            self.markInvalid()
            return

        val = input_node.getOutput(other_index)
        print("val:", val)
        print("inputnode:", input_node)
        if val is None:
            self.grNode.setToolTip("Input is NaN")
            self.markInvalid()
            return

        self.content.edit.setText(val)
        self.markInvalid(False)
        self.markDirty(False)
        self.grNode.setToolTip("")


        self.content.update()
        self.grNode.update()

        self.markChildrenDirty(True)

        return val
    def onInputChanged(self, socket=None):
        print("%s::__onInputChanged" % self.__class__.__name__)
        self.markDirty()
        self.eval()
