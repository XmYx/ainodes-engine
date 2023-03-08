from qtpy.QtWidgets import QLabel
from qtpy.QtCore import Qt
from ainodes_frontend.nodes.base.node_config import register_node, OP_NODE_IMG_PREVIEW
from ainodes_frontend.nodes.base.ai_node_base import CalcNode, CalcGraphicsNode
from ainodes_backend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_backend.node_engine.utils import dumpException
from qtpy import QtWidgets

class ImageOutputWidget(QDMNodeContentWidget):
    def initUI(self):
        self.image = QLabel(self)
        self.image.setAlignment(Qt.AlignRight)
        self.image.setObjectName(self.node.content_label_objname)
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(15, 30, 15, 35)
        layout.addWidget(self.image)
        self.setLayout(layout)


    def serialize(self):
        res = super().serialize()
        #res['value'] = self.edit.text()
        return res

    def deserialize(self, data, hashmap={}):
        res = super().deserialize(data, hashmap)
        try:
            #value = data['value']
            #self.image.setPixmap(value)
            return True & res
        except Exception as e:
            dumpException(e)
        return res


@register_node(OP_NODE_IMG_PREVIEW)
class ImagePreviewWidget(CalcNode):
    icon = "icons/out.png"
    op_code = OP_NODE_IMG_PREVIEW
    op_title = "Image Preview"
    content_label_objname = "image_output_node"
    category = "image"


    def __init__(self, scene):
        super().__init__(scene, inputs=[5,1], outputs=[5,1])
        #self.eval()
        #self.content.eval_signal.connect(self.evalImplementation)

    def initInnerClasses(self):
        self.content = ImageOutputWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.output_socket_name = ["EXEC", "IMAGE"]
        self.input_socket_name = ["EXEC", "IMAGE"]

        #self.content.mark_dirty_signal.connect(self.markDirty)
        #self.content.image.changeEvent.connect(self.onInputChanged)

    def evalImplementation(self, index=0):
        #self.markDirty(True)
        if self.getInput(0) is not None:
            input_node, other_index = self.getInput(0)
            if not input_node:
                self.grNode.setToolTip("Input is not connected")
                self.markInvalid()
                return

            val = input_node.getOutput(other_index)

            if val is None:
                self.grNode.setToolTip("Input is NaN")
                self.markInvalid()
                return
            #print("Preview Node Value", val)
            self.content.image.setPixmap(val)
            self.setOutput(0, val)
            self.markInvalid(False)
            self.markDirty(False)
            self.grNode.setToolTip("")
            self.grNode.height = val.height() + 96
            self.grNode.width = val.width() + 32
            self.content.image.setMinimumHeight(self.content.image.pixmap().size().height())
            self.content.image.setMinimumWidth(self.content.image.pixmap().size().width())
            self.content.setGeometry(0,0,self.content.image.pixmap().size().width(), self.content.image.pixmap().size().height())
            for socket in self.outputs + self.inputs:
                socket.setSocketPosition()
            self.updateConnectedEdges()
            #print("Reloaded")
            if len(self.getOutputs(1)) > 0:
                self.executeChild(output_index=1)
        else:
            val = self.value
        return val

    def onInputChanged(self, socket=None):
        #super().onInputChanged(socket=socket)
        self.markDirty(True)
        self.markInvalid(True)
        self.eval()