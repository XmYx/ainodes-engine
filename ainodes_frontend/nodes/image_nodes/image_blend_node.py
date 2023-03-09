import cv2
import numpy as np
from PIL import ImageOps, Image
#from qtpy.QtWidgets import QLineEdit, QLabel, QPushButton, QFileDialog, QVBoxLayout
from qtpy import QtWidgets, QtCore

from ainodes_backend.cnet_preprocessors import hed
from ainodes_backend.cnet_preprocessors.mlsd import MLSDdetector

from ainodes_backend.cnet_preprocessors.midas import MidasDetector
from ainodes_backend.cnet_preprocessors.openpose import OpenposeDetector
from ainodes_frontend.nodes.base.node_config import register_node, OP_NODE_IMAGE_BLEND
from ainodes_frontend.nodes.base.ai_node_base import CalcNode, CalcGraphicsNode
from ainodes_backend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_backend.node_engine.utils import dumpException
from ainodes_frontend.nodes.qops.qimage_ops import pixmap_to_pil_image, pil_image_to_pixmap

class BlendWidget(QDMNodeContentWidget):
    def initUI(self):
        # Create a label to display the image
        self.text_label = QtWidgets.QLabel("Image Operator:")

        self.blend = QtWidgets.QDoubleSpinBox()
        self.blend.setMinimum(0.00)
        self.blend.setSingleStep(0.01)
        self.blend.setMaximum(1.00)
        self.blend.setValue(0.00)


        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self.blend)

        self.setLayout(layout)


    def serialize(self):
        res = super().serialize()
        res['blend'] = self.blend.value()
        return res

    def deserialize(self, data, hashmap={}):
        res = super().deserialize(data, hashmap)
        try:
            self.blend.setValue(int(data['h']))
            #self.image.setPixmap(value)
            return True & res
        except Exception as e:
            dumpException(e)
        return res


@register_node(OP_NODE_IMAGE_BLEND)
class BlendNode(CalcNode):
    icon = "icons/in.png"
    op_code = OP_NODE_IMAGE_BLEND
    op_title = "Image Blend"
    content_label_objname = "image_blend_node"
    category = "image"


    def __init__(self, scene):
        super().__init__(scene, inputs=[5,5,1], outputs=[5,1])
        #self.eval()
        #self.content.eval_signal.connect(self.eval)

    def initInnerClasses(self):
        self.content = BlendWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.output_socket_name = ["EXEC", "IMAGE"]
        self.input_socket_name = ["EXEC", "IMAGE1", "IMAGE2"]

        self.grNode.height = 200
    @QtCore.Slot(int)
    def evalImplementation(self, index=0):
        if self.getInput(1) != None:
            node, index = self.getInput(1)
            pixmap1 = node.getOutput(index)
        else:
            pixmap1 = None
        if self.getInput(0) != None:
            node, index = self.getInput(0)
            pixmap2 = node.getOutput(index)
        else:
            pixmap2 = None

        if pixmap1 != None and pixmap2 != None:
            blend = self.content.blend.value()
            self.value = self.image_op(pixmap1, pixmap2, blend)
            print(f"BLEND NODE: Using both inputs with a blend value: {blend}")

            self.setOutput(0, self.value)
            self.markDirty(False)
            self.markInvalid(False)
        if pixmap1 != None:
            try:
                self.setOutput(0, pixmap2)
                print(f"BLEND NODE: Using only Second input")

            except:
                pass
        elif pixmap2 != None:
            try:
                self.setOutput(0, pixmap1)
                print(f"BLEND NODE: Using only First input")
            except:
                pass
        if len(self.getOutputs(1)) > 0:
            self.executeChild(output_index=1)
        return self.value
    def onMarkedDirty(self):
        self.value = None
    def eval(self):
        self.markDirty(True)
        self.evalImplementation()
    def image_op(self, pixmap1, pixmap2, blend):
        # Convert the QPixmap object to a PIL Image object
        image1 = pixmap_to_pil_image(pixmap1).convert("RGBA")
        image2 = pixmap_to_pil_image(pixmap2).convert("RGBA")

        image = Image.blend(image1, image2, blend)
        #print(blend, image)

        # Convert the PIL Image object to a QPixmap object
        pixmap = pil_image_to_pixmap(image)

        return pixmap

