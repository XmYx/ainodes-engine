import threading
import torch
from PIL import Image
from PIL.ImageQt import ImageQt
from qtpy import QtWidgets, QtCore, QtGui
from qtpy.QtGui import QPixmap

from ainodes_backend.torch_gc import torch_gc
from ainodes_frontend.nodes.base.node_config import register_node, get_next_opcode
from ainodes_frontend.nodes.base.ai_node_base import CalcNode, CalcGraphicsNode
from ainodes_backend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_backend.node_engine.utils import dumpException
from ainodes_backend import singleton as gs

OP_NODE_EXAMPLE = get_next_opcode()


class ExampleWidget(QDMNodeContentWidget):
    def initUI(self):
        self.label = QtWidgets.QLabel("Label:")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(15,15,15,25)
        layout.addWidget(self.label)

        self.setLayout(layout)


    def serialize(self):
        res = super().serialize()
        res['label'] = self.label.currentText()
        return res

    def deserialize(self, data, hashmap={}):
        res = super().deserialize(data, hashmap)
        try:
            self.label.setCurrentText(data['label'])
            return True & res
        except Exception as e:
            dumpException(e)
        return res

@register_node(OP_NODE_EXAMPLE)
class ExampleNode(CalcNode):
    icon = "icons/in.png"
    op_code = OP_NODE_EXAMPLE
    op_title = "Example"
    content_label_objname = "example_node"
    category = "debug"
    def __init__(self, scene):
        super().__init__(scene, inputs=[2,3,3,1], outputs=[5,2,1])
        self.content.button.clicked.connect(self.evalImplementation)
        self.busy = False
        # Create a worker object
    def initInnerClasses(self):
        self.content = ExampleWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.height = 500
        self.grNode.width = 256
        self.content.setMinimumWidth(256)
        self.content.setMinimumHeight(256)
        self.input_socket_name = ["EXEC", "VALUE"]
        self.output_socket_name = ["EXEC", "VALUE"]

    def evalImplementation(self, index=0):
        self.markDirty(True)
        if self.value is None:
            # Start the worker thread
            if self.busy == False:
                self.busy = True
                thread0 = threading.Thread(target=self.node_thread, args=())
                thread0.start()
            return None
        else:
            self.markDirty(False)
            self.markInvalid(False)
            return self.value

    def onMarkedDirty(self):
        self.value = None
    def node_thread(self, progress_callback=None):
        try:
            connected_input_node, index = self.getInput(1)
            value = connected_input_node.getOutput(index)
        except Exception as e:
            print(e)
            value = None
        try:

            """Ideally do all processing in this, safe, threaded function, 
            unless you want to display / change something in the gui, 
            then you should do it in the onWorkerFinished function"""

            result = None
            self.onWorkerFinished(result)
        except:
            self.busy = False
            if len(self.getOutputs(2)) > 0:
                self.executeChild(output_index=2)
        return result
    def onWorkerFinished(self, result):
        self.markDirty(False)
        self.markInvalid(False)
        self.setOutput(0, result)
        self.busy = False
        if len(self.getOutputs(1)) > 0:
            self.executeChild(output_index=1)
        return
    def onInputChanged(self, socket=None):
        pass

