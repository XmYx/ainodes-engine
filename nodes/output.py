from PIL import Image
from PIL.ImageQt import ImageQt
from qtpy.QtWidgets import QLineEdit, QLabel
from qtpy.QtCore import Qt
from qtpy.QtGui import QPixmap
from nodes.base.calc_conf import register_node, OP_NODE_IMG_PREVIEW
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


@register_node(OP_NODE_IMG_PREVIEW)
class ImageInputNode(CalcNode):
    icon = "icons/out.png"
    op_code = OP_NODE_IMG_PREVIEW
    op_title = "Output"
    content_label_objname = "image_output_node"

    def __init__(self, scene):
        super().__init__(scene, inputs=[1], outputs=[1])
        self.eval()
        self.content.eval_signal.connect(self.eval)

    def initInnerClasses(self):
        self.content = ImageOutputWidget(self)
        self.grNode = CalcGraphicsNode(self)

        #self.content.image.changeEvent.connect(self.onInputChanged)

    def evalImplementation(self, index=0):
        if self.getInput(0) != None:
            input_node, other_index = self.getInput(0)
            if not input_node:
                self.grNode.setToolTip("Input is not connected")
                self.markInvalid()
                return

            val = input_node.eval()

            if val is None:
                self.grNode.setToolTip("Input is NaN")
                self.markInvalid()
                return

            self.content.image.setPixmap(val)
            self.markInvalid(False)
            self.markDirty(False)
            self.grNode.setToolTip("")

            self.grNode.height = self.content.image.pixmap().size().height() + 32
            self.grNode.width = self.content.image.pixmap().size().width() + 32
            self.content.image.setMinimumHeight(self.content.image.pixmap().size().height())
            self.content.image.setMinimumWidth(self.content.image.pixmap().size().width())
            self.content.setMinimumHeight(self.content.image.pixmap().size().height())
            self.content.setMinimumWidth(self.content.image.pixmap().size().width())

            self.content.update()
            self.grNode.update()

            self.markChildrenDirty(True)

            return val
        else:
            self.markChildrenDirty(True)
            self.evalChildren()
            return self.value

