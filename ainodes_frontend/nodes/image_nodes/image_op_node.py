import cv2
import numpy as np
from PIL import ImageOps, Image
#from qtpy.QtWidgets import QLineEdit, QLabel, QPushButton, QFileDialog, QVBoxLayout
from qtpy import QtWidgets, QtCore

from ainodes_backend.cnet_preprocessors import hed
from ainodes_backend.cnet_preprocessors.mlsd import MLSDdetector

from ainodes_backend.cnet_preprocessors.midas import MidasDetector
from ainodes_backend.cnet_preprocessors.openpose import OpenposeDetector
from ainodes_frontend.nodes.base.node_config import register_node, OP_NODE_IMAGE_OPS
from ainodes_frontend.nodes.base.ai_node_base import CalcNode, CalcGraphicsNode
from ainodes_backend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_backend.node_engine.utils import dumpException
from ainodes_frontend.nodes.qops.qimage_ops import pixmap_to_pil_image, pil_image_to_pixmap

image_ops_methods = [
    "resize",
    "canny",
    "fake_scribble",
    'hed',
    'depth',
    'normal',
    'mlsd',
    'openpose',
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

        self.midas_a = QtWidgets.QDoubleSpinBox()
        self.midas_a.setMinimum(0)
        self.midas_a.setSingleStep(0.01)
        self.midas_a.setMaximum(100)
        self.midas_a.setValue(np.pi*2.0)
        self.midas_a.setVisible(False)

        self.midas_bg = QtWidgets.QDoubleSpinBox()
        self.midas_bg.setMinimum(0)
        self.midas_bg.setSingleStep(1)
        self.midas_bg.setMaximum(100)
        self.midas_bg.setValue(0.01)
        self.midas_bg.setVisible(False)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self.dropdown)
        layout.addWidget(self.width_value)
        layout.addWidget(self.height_value)
        layout.addWidget(self.canny_low)
        layout.addWidget(self.canny_high)
        layout.addWidget(self.midas_a)
        layout.addWidget(self.midas_bg)

        self.setLayout(layout)

    def dropdownChanged(self, event):
        value = self.dropdown.currentText()
        if value == 'resize':
            self.width_value.setVisible(True)
            self.height_value.setVisible(True)
            self.canny_high.setVisible(False)
            self.canny_low.setVisible(False)
            self.midas_a.setVisible(False)
            self.midas_bg.setVisible(False)

        elif value == 'canny':
            self.width_value.setVisible(False)
            self.height_value.setVisible(False)
            self.canny_high.setVisible(True)
            self.canny_low.setVisible(True)
            self.midas_a.setVisible(False)
            self.midas_bg.setVisible(False)
        elif value in ['depth', 'normal', 'mlsd']:
            self.width_value.setVisible(False)
            self.height_value.setVisible(False)
            self.canny_high.setVisible(False)
            self.canny_low.setVisible(False)
            self.midas_a.setVisible(True)
            self.midas_bg.setVisible(True)

        else:
            self.width_value.setVisible(False)
            self.height_value.setVisible(False)
            self.canny_high.setVisible(False)
            self.canny_low.setVisible(False)
            self.midas_a.setVisible(False)
            self.midas_bg.setVisible(False)

    def serialize(self):
        res = super().serialize()
        res['dropdown'] = self.dropdown.currentText()
        res['w'] = self.width_value.value()
        res['h'] = self.height_value.value()
        res['canny_high'] = self.canny_high.value()
        res['canny_low'] = self.canny_low.value()
        res['midas_a'] = self.midas_a.value()
        res['midas_bg'] = self.midas_bg.value()
        return res

    def deserialize(self, data, hashmap={}):
        res = super().deserialize(data, hashmap)
        try:
            self.dropdown.setCurrentText(data['dropdown'])
            self.height_value.setValue(int(data['h']))
            self.width_value.setValue(int(data['w']))
            self.canny_high.setValue(float(data['canny_high']))
            self.canny_low.setValue(float(data['canny_low']))
            self.midas_a.setValue(float(data['midas_a']))
            self.midas_bg.setValue(float(data['midas_bg']))
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
        super().__init__(scene, inputs=[5,1], outputs=[5,1])
        #self.eval()
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
        #if self.getInput(index) != None:
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

        elif method == 'fake_scribble':
            image = np.array(image)
            detector = hed.HEDdetector()
            image = detector(image)
            image = HWC3(image)
            image = hed.nms(image, 127, 3.0)
            image = cv2.GaussianBlur(image, (0, 0), 3.0)
            image[image > 4] = 255
            image[image < 255] = 0
            image = Image.fromarray(image)
            del detector
        elif method == 'hed':
            # Ref: https://github.com/lllyasviel/ControlNet/blob/main/gradio_hed2image.py
            image = np.array(image)
            detector = hed.HEDdetector()
            image = detector(image)
            image = HWC3(image)
            image = Image.fromarray(image)
            del detector
        elif method == 'depth':
            image = np.array(image)
            detector = MidasDetector()
            a = self.content.midas_a.value()
            bg_threshold = self.content.midas_bg.value()
            depth_map_np, normal_map_np = detector(image, a, bg_threshold)
            image = HWC3(depth_map_np)
            image = Image.fromarray(image)
            del detector
        elif method == 'normal':
            image = np.array(image)
            detector = MidasDetector()
            a = self.content.midas_a.value()
            bg_threshold = self.content.midas_bg.value()
            depth_map_np, normal_map_np = detector(image, a, bg_threshold)
            image = HWC3(normal_map_np)
            image = Image.fromarray(image)
            del detector
        elif method == 'mlsd':
            image = image.convert('RGB')
            image = np.array(image)
            print(image.shape)
            detector = MLSDdetector()
            a = self.content.midas_a.value()
            bg_threshold = self.content.midas_bg.value()
            mlsd = detector(image, bg_threshold, a)
            image = HWC3(mlsd)
            image = Image.fromarray(image)
            del detector
        elif method == 'openpose':
            image = image.convert('RGB')
            image = np.array(image)
            print(image.shape)
            detector = OpenposeDetector()
            pose, _ = detector(image, True)
            #image = HWC3(pose)
            image = Image.fromarray(pose)
            del detector

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
