import cv2
import numpy as np
from PIL import ImageOps, Image
#from qtpy.QtWidgets import QLineEdit, QLabel, QPushButton, QFileDialog, QVBoxLayout
from qtpy import QtWidgets, QtCore
from nodes.base.node_config import register_node, OP_NODE_IMAGE_OPS
from nodes.base.ai_node_base import CalcNode, CalcGraphicsNode
from node_engine.node_content_widget import QDMNodeContentWidget
from node_engine.utils import dumpException
from nodes.qops.qimage_ops import pixmap_to_pil_image, pil_image_to_pixmap
import backend.singleton as gs


image_ops_methods = [
    "resize",
    "canny",
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
image_ops_valid_methods = [
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
        self.dropdown.currentIndexChanged.connect(self.dropdownChanged)

        self.width_value = QtWidgets.QSpinBox()
        self.width_value.setMinimum(64)
        self.width_value.setSingleStep(64)
        self.width_value.setMaximum(4096)
        self.width_value.setValue(512)

        self.height_value = QtWidgets.QSpinBox()
        self.height_value.setMinimum(64)
        self.height_value.setSingleStep(64)
        self.height_value.setMaximum(4096)
        self.height_value.setValue(512)

        self.canny_low = QtWidgets.QSpinBox()
        self.canny_low.setMinimum(0)
        self.canny_low.setSingleStep(1)
        self.canny_low.setMaximum(255)
        self.canny_low.setValue(100)
        self.canny_low.setVisible(False)

        self.canny_high = QtWidgets.QSpinBox()
        self.canny_high.setMinimum(0)
        self.canny_high.setSingleStep(1)
        self.canny_high.setMaximum(255)
        self.canny_high.setValue(100)
        self.canny_high.setVisible(False)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self.dropdown)
        layout.addWidget(self.width_value)
        layout.addWidget(self.height_value)
        layout.addWidget(self.canny_low)
        layout.addWidget(self.canny_high)
        self.setLayout(layout)

    def dropdownChanged(self, event):
        value = self.dropdown.currentText()
        if value == 'resize':
            self.width_value.setVisible(True)
            self.height_value.setVisible(True)
            self.canny_high.setVisible(False)
            self.canny_low.setVisible(False)

        elif value == 'canny':
            self.width_value.setVisible(False)
            self.height_value.setVisible(False)
            self.canny_high.setVisible(True)
            self.canny_low.setVisible(True)
        else:
            self.width_value.setVisible(False)
            self.height_value.setVisible(False)
            self.canny_high.setVisible(False)
            self.canny_low.setVisible(False)
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
    category = "image"


    def __init__(self, scene):
        super().__init__(scene, inputs=[3,3], outputs=[3,3])
        self.eval()
        self.content.eval_signal.connect(self.eval)

    def initInnerClasses(self):
        self.content = ImageOpsWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.content.dropdown.currentIndexChanged.connect(self.evalImplementation)
        self.output_socket_name = ["EXEC", "IMAGE"]
        self.input_socket_name = ["EXEC", "IMAGE"]

        self.grNode.height = 200
        #self.grNode.width = 256
        #self.content.setMinimumHeight(400)
        #self.content.setMinimumWidth(256)

        #self.content.image.changeEvent.connect(self.onInputChanged)
    @QtCore.Slot(int)
    def evalImplementation(self, index=0):
        self.value = None
        if self.value == None:
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
                if len(self.getOutputs(1)) > 0:
                    self.executeChild(output_index=1)
                return self.value
        else:
            self.markDirty(False)
            self.markInvalid(False)
            self.markDescendantsDirty()
            self.evalChildren()
            return self.value

    def onMarkedDirty(self):
        self.value = None
    def eval(self):
        self.markDirty(True)
        self.evalImplementation()
    def image_op(self, pixmap, method):
        # Convert the QPixmap object to a PIL Image object
        image = pixmap_to_pil_image(pixmap)
        if method in image_ops_valid_methods:
            # Get the requested ImageOps method
            ops_method = getattr(ImageOps, method, None)

            if ops_method:
                # Apply the ImageOps method to the PIL Image object
                image = ops_method(image)
            else:
                # If the requested method is not available, raise an error
                raise ValueError(f"Invalid ImageOps method: {method}")
        elif method == 'resize':
            width = self.content.width_value.value()
            height = self.content.height_value.value()
            image = image.resize((width, height), resample=Image.LANCZOS)
        elif method == 'canny':

            image = np.array(image)
            image = cv2.Canny(image, self.content.canny_low.value(), self.content.canny_high.value(), L2gradient=True)
            image = HWC3(image)
            image = Image.fromarray(image)

        # Convert the PIL Image object to a QPixmap object
        pixmap = pil_image_to_pixmap(image)

        return pixmap

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y
