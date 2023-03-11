from qtpy.QtWidgets import QVBoxLayout
from ainodes_frontend.nodes.base.node_config import register_node, get_next_opcode
from ainodes_frontend.nodes.base.ai_node_base import CalcNode, CalcGraphicsNode
from ainodes_backend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_backend.node_engine.utils import dumpException

OP_NODE_DEBUG_MULTI_INPUT = get_next_opcode()
class ImageInputWidget(QDMNodeContentWidget):
    def initUI(self):
        # Create a label to display the image
        layout = QVBoxLayout(self)
        self.setLayout(layout)


    def serialize(self):
        res = super().serialize()
        #res['value'] = self.edit.text()
        return res

    def deserialize(self, data, hashmap={}):
        res = super().deserialize(data, hashmap)
        try:
            value = data['value']
            return True & res
        except Exception as e:
            dumpException(e)
        return res


@register_node(OP_NODE_DEBUG_MULTI_INPUT)
class ImageInputNode(CalcNode):
    icon = "icons/in.png"
    op_code = OP_NODE_DEBUG_MULTI_INPUT
    op_title = "MultiInput"
    content_label_objname = "debug_multi_input_node"
    category = "debug"

    def __init__(self, scene):
        super().__init__(scene, inputs=[], outputs=[3,3,3])

        self.setOutput(0, "Test1")
        self.setOutput(1, "Test2")
        self.setOutput(2, "Test3")

        self.eval(0)

    def initInnerClasses(self):
        self.content = ImageInputWidget(self)
        self.grNode = CalcGraphicsNode(self)

    def evalImplementation(self, index=0):
        u_value = self.getOutput(index)
        print("u_value:", u_value)
        self.markDirty(False)
        self.markInvalid(False)
        self.markDescendantsInvalid(False)
        #self.markDescendantsDirty()
        #self.grNode.setToolTip("")
        #self.evalChildren()
        return u_value