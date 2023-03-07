from PIL.ImageQt import ImageQt
#from qtpy.QtWidgets import QLineEdit, QLabel, QPushButton, QFileDialog, QVBoxLayout
from qtpy import QtWidgets, QtCore
from qtpy.QtGui import QPixmap
from ainodes_frontend.nodes.base.node_config import register_node, OP_NODE_DIFFUSERS_SAMPLER
from ainodes_frontend.nodes.base.ai_node_base import CalcNode, CalcGraphicsNode
from ainodes_backend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_backend.node_engine.utils import dumpException
import ainodes_backend.singleton as gs
from ainodes_backend.worker.worker import Worker


class DiffusersWidget(QDMNodeContentWidget):
    def initUI(self):
        # Create a label to display the image
        self.text_label = QtWidgets.QLabel("Diffusers:")
        self.prompt = QtWidgets.QTextEdit()
        self.steps = QtWidgets.QSpinBox()
        self.steps.setMinimum(1)
        self.steps.setMaximum(1000)
        self.steps.setValue(25)
        self.button = QtWidgets.QPushButton("Load diffusers")
        #self.button.clicked.connect(self.parent.parent.load_diffusers)
        self.infer_button = QtWidgets.QPushButton("Infer Diffusers")
        #self.infer_button.clicked.connect(self.parent.parent.emit_run_signal)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self.text_label)
        layout.addWidget(self.prompt)
        layout.addWidget(self.steps)
        layout.addWidget(self.button)
        layout.addWidget(self.infer_button)

        self.setLayout(layout)


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


@register_node(OP_NODE_DIFFUSERS_SAMPLER)
class DiffusersNode(CalcNode):
    icon = "icons/in.png"
    op_code = OP_NODE_DIFFUSERS_SAMPLER
    op_title = "Diffusers Sampling"
    content_label_objname = "diffusers_sampling_node"
    category = "sampling"


    def __init__(self, scene):
        super().__init__(scene, inputs=[3,3], outputs=[3])
        self.eval()
        self.content.eval_signal.connect(self.eval)
        # Create a worker object
    def initInnerClasses(self):
        self.content = DiffusersWidget(self)
        self.grNode = CalcGraphicsNode(self)

        self.grNode.height = 480
        self.grNode.width = 256
        self.content.setMinimumHeight(400)
        self.content.setMinimumWidth(256)

        #self.content.image.changeEvent.connect(self.onInputChanged)

    def evalImplementation(self, index=0):

        if self.value is None:
            # Start the worker thread
            self.worker = Worker(self.infer_diffusers)
            # Connect the worker's finished signal to a slot that updates the node value
            self.worker.signals.result.connect(self.onWorkerFinished)
            self.scene.threadpool.start(self.worker)
            #self.worker.run()
            # Return None as a placeholder value
            return None
        else:
            self.markDirty(False)
            self.markInvalid(False)
            self.executeChild()
            #self.markDescendantsDirty()
            #self.evalChildren()
            return self.value

    def onMarkedDirty(self):
        self.value = None
    def infer_diffusers(self, progress_callback=None):
        node, index = self.getInput(0)
        pipe = node.eval(index)
        image = gs.obj[pipe](prompt=self.content.prompt.toPlainText(), num_inference_steps=self.content.steps.value()).images[0]
        qimage = ImageQt(image)
        pixmap = QPixmap().fromImage(qimage)
        return pixmap
    @QtCore.Slot(object)
    def onWorkerFinished(self, result):
        # Update the node value and mark it as dirty
        self.value = result
        self.setOutput(0, result)
        self.markDirty(False)
        self.markInvalid(False)
        #self.markDescendantsDirty()
        #self.evalChildren()