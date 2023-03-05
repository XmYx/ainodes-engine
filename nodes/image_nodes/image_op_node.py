import torch
from PIL import Image, ImageOps, ImageChops
from PIL.ImageQt import ImageQt
from diffusers import StableDiffusionPipeline
#from qtpy.QtWidgets import QLineEdit, QLabel, QPushButton, QFileDialog, QVBoxLayout
from qtpy import QtWidgets, QtCore
from qtpy.QtCore import Qt
from qtpy.QtGui import QPixmap
from nodes.base.node_config import register_node, OP_NODE_IMAGE_OPS
from nodes.base.ai_node_base import CalcNode, CalcGraphicsNode
from node_engine.node_content_widget import QDMNodeContentWidget
from node_engine.utils import dumpException
from nodes.qops.qimage_ops import pixmap_to_pil_image, pil_image_to_pixmap
from singleton import Singleton

gs = Singleton()

image_ops_methods = [
    "autocontrast",
    "colorize",
    "contrast",
    "grayscale",
    "invert",
    "mirror",
    "posterize",
    "solarize",
    "flip"
]

class ImageOpsWidget(QDMNodeContentWidget):
    def initUI(self):
        # Create a label to display the image
        self.text_label = QtWidgets.QLabel("Image Operator:")
        self.dropdown = QtWidgets.QComboBox()
        self.dropdown.addItems(image_ops_methods)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self.dropdown)
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


@register_node(OP_NODE_IMAGE_OPS)
class ImageOpNode(CalcNode):
    icon = "icons/in.png"
    op_code = OP_NODE_IMAGE_OPS
    op_title = "Image Operators"
    content_label_objname = "diffusers_sampling_node"

    def __init__(self, scene):
        super().__init__(scene, inputs=[3], outputs=[3])
        self.eval()
        self.content.eval_signal.connect(self.eval)

    def initInnerClasses(self):
        self.content = ImageOpsWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.content.dropdown.currentIndexChanged.connect(self.evalImplementation)

        #self.grNode.height = 480
        #self.grNode.width = 256
        #self.content.setMinimumHeight(400)
        #self.content.setMinimumWidth(256)

        #self.content.image.changeEvent.connect(self.onInputChanged)
    @QtCore.Slot(int)
    def evalImplementation(self, index=0):

        if self.isDirty() == True:
            if self.getInput(index) != None:
                #self.markInvalid()
                #self.markDescendantsDirty()
                node, index = self.getInput(index)

                print("RETURN", node, index)

                pixmap = node.getOutput(index)
                method = self.content.dropdown.currentText()
                self.value = self.image_op(pixmap, method)
                self.setOutput(0, self.value)
                self.markDirty(False)
                self.markInvalid(False)
                self.evalChildren()
                return self.value
        else:
            self.markDirty(False)
            self.markInvalid(False)
            self.markDescendantsDirty()
            self.evalChildren()
            return self.value

    def onMarkedDirty(self):
        self.value = None

    def image_op(self, pixmap, method):
        # Convert the QPixmap object to a PIL Image object
        image = pixmap_to_pil_image(pixmap)

        # Get the requested ImageOps method
        ops_method = getattr(ImageOps, method, None)

        if ops_method:
            # Apply the ImageOps method to the PIL Image object
            image = ops_method(image)
        else:
            # If the requested method is not available, raise an error
            raise ValueError(f"Invalid ImageOps method: {method}")

        # Convert the PIL Image object to a QPixmap object
        pixmap = pil_image_to_pixmap(image)

        return pixmap

