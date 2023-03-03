from PIL import Image
from PIL.ImageQt import ImageQt
from qtpy.QtWidgets import QLineEdit, QLabel
from qtpy.QtCore import Qt
from qtpy.QtGui import QPixmap
from nodes.base.calc_conf import register_node, OP_NODE_LOOP_NODE
from nodes.base.calc_node_base import CalcNode, CalcGraphicsNode
from nodeeditor.node_content_widget import QDMNodeContentWidget
from nodeeditor.utils import dumpException


class ImageOutputWidget(QDMNodeContentWidget):
    def initUI(self):
        #self.edit = QLineEdit("1", self)
        #self.edit.setAlignment(Qt.AlignRight)
        #self.edit.setObjectName(self.node.content_label_objname)
        self.image = QLabel("2", self)
        self.image.setObjectName(self.node.content_label_objname)
    def serialize(self):
        res = super().serialize()
        #res['value'] = self.edit.text()
        return res

    def deserialize(self, data, hashmap={}):
        res = super().deserialize(data, hashmap)
        try:
            value = data['value']
            self.image.setPixmap(value)
            return True & res
        except Exception as e:
            dumpException(e)
        return res


@register_node(OP_NODE_LOOP_NODE)
class LoopNode(CalcNode):
    icon = "icons/out.png"
    op_code = OP_NODE_LOOP_NODE
    op_title = "Loop"
    content_label_objname = "image_output_node"

    def __init__(self, scene):
        super().__init__(scene, inputs=[1], outputs=[1])
        self.eval()
        self.content.eval_signal.connect(self.eval)

    def initInnerClasses(self):
        self.content = ImageOutputWidget(self)
        self.grNode = CalcGraphicsNode(self)

    def evalImplementation(self, index=0):
        node, index = self.getInput(0)
        node.markDirty()
        self.markChildrenDirty()
        for node in self.getChildrenNodes():
            node.markDirty()
            node.content.eval_signal.emit(0)

    def onInputChanged(self, socket: 'Socket'):
        """Event handling when Node's input Edge has changed. We auto-mark this `Node` to be `Dirty` with all it's
        descendants

        :param socket: reference to the changed :class:`~nodeeditor.node_socket.Socket`
        :type socket: :class:`~nodeeditor.node_socket.Socket`
        """
        self.eval()
        socket.node.markDirty()
        self.evalChildren()
        #self.markDirty()
        #self.markDescendantsDirty()
        #for socket in self.outputs:
        #    socket.node.markDirty()
        #    socket.node.content.eval_signal.emit(0)