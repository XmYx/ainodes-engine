import torch
from PIL import Image, ImageOps, ImageChops
from PIL.ImageQt import ImageQt
from diffusers import StableDiffusionPipeline
#from qtpy.QtWidgets import QLineEdit, QLabel, QPushButton, QFileDialog, QVBoxLayout
from qtpy import QtWidgets, QtCore
from qtpy.QtCore import Qt
from qtpy.QtGui import QPixmap
from nodes.base.node_config import register_node, OP_NODE_LATENT
from nodes.base.ai_node_base import CalcNode, CalcGraphicsNode
from node_engine.node_content_widget import QDMNodeContentWidget
from node_engine.utils import dumpException
from nodes.qops.qimage_ops import pixmap_to_pil_image, pil_image_to_pixmap
import singleton as gs

class LatentWidget(QDMNodeContentWidget):
    def initUI(self):
        # Create a label to display the image
        self.width = QtWidgets.QSpinBox()
        self.width.setMinimum(64)
        self.width.setMaximum(4096)
        self.width.setValue(64)
        self.width.setSingleStep(64)

        self.height = QtWidgets.QSpinBox()
        self.height.setMinimum(64)
        self.height.setMaximum(4096)
        self.height.setValue(64)
        self.height.setSingleStep(64)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(15,15,15,20)
        layout.setSpacing(10)
        layout.addWidget(self.width)
        layout.addWidget(self.height)
        self.setLayout(layout)

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


@register_node(OP_NODE_LATENT)
class LatentNode(CalcNode):
    icon = "icons/in.png"
    op_code = OP_NODE_LATENT
    op_title = "Image Operators"
    content_label_objname = "diffusers_sampling_node"

    def __init__(self, scene):
        super().__init__(scene, inputs=[3], outputs=[3,3])
        self.eval()
        self.content.eval_signal.connect(self.eval)

    def initInnerClasses(self):
        self.content = LatentWidget(self)
        self.grNode = CalcGraphicsNode(self)
        #self.content.dropdown.currentIndexChanged.connect(self.evalImplementation)
        self.input_socket_name = ["EXEC"]
        self.output_socket_name = ["EXEC", "LATENT"]

        self.grNode.height = 160
        self.grNode.width = 200
        #self.content.setMinimumHeight(400)
        #self.content.setMinimumWidth(256)

        #self.content.image.changeEvent.connect(self.onInputChanged)
    @QtCore.Slot(int)
    def evalImplementation(self, index=0):

        if self.isDirty() == True:
            if self.getInput(index) != None:
                #self.markInvalid()
                #self.markDescendantsDirty()
                self.value = self.generate_latent()
                self.setOutput(0, self.value)
                self.markDirty(False)
                self.markInvalid(False)
                self.executeChild(output_index=1)
                return self.value
        else:
            self.markDirty(False)
            self.markInvalid(False)
            #self.markDescendantsDirty()
            if len(self.getOutputs(1)) > 0:
                self.executeChild(output_index=1)
            return self.value

    def onMarkedDirty(self):
        self.value = None

    def generate_latent(self):
        width = self.content.width.value()
        height = self.content.height.value()
        batch_size = 1
        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
        return latent

