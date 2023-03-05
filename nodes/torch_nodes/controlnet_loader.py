import os

import torch
from PIL import Image
from PIL.ImageQt import ImageQt
from diffusers import StableDiffusionPipeline
#from qtpy.QtWidgets import QLineEdit, QLabel, QPushButton, QFileDialog, QVBoxLayout
from qtpy import QtWidgets
from qtpy.QtCore import Qt
from qtpy.QtGui import QPixmap

from backend.model_loader import ModelLoader
from backend.controlnet_loader import load_controlnet
from nodes.base.node_config import register_node, OP_NODE_CONTROLNET_LOADER
from nodes.base.ai_node_base import CalcNode, CalcGraphicsNode
from node_engine.node_content_widget import QDMNodeContentWidget
from node_engine.utils import dumpException
#from singleton import Singleton

#gs = Singleton()

import singleton
gs = singleton
class ControlnetLoaderWidget(QDMNodeContentWidget):
    def initUI(self):
        # Create a label to display the image
        # Create the dropdown widget
        self.control_net_name = QtWidgets.QComboBox(self)
        #self.dropdown.currentIndexChanged.connect(self.on_dropdown_changed)
        # Populate the dropdown with .ckpt and .safetensors files in the checkpoints folder
        checkpoint_folder = "comfy_defaults/models/controlnet"
        checkpoint_files = [f for f in os.listdir(checkpoint_folder) if f.endswith(('.ckpt', '.pt', '.bin', '.pth', ".safetensors"))]
        self.control_net_name.addItems(checkpoint_files)
        # Add the dropdown widget to the layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.control_net_name)
        self.setLayout(layout)
        self.setSizePolicy(CenterExpandingSizePolicy(self))
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
class CenterExpandingSizePolicy(QtWidgets.QSizePolicy):
    def __init__(self, parent=None):
        super().__init__(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.parent = parent
        self.setHorizontalStretch(0)
        self.setVerticalStretch(0)
        self.setRetainSizeWhenHidden(True)
        self.setHorizontalPolicy(QtWidgets.QSizePolicy.Expanding)
        self.setVerticalPolicy(QtWidgets.QSizePolicy.Expanding)
        #self.parent.setAlignment(Qt.AlignCenter)

@register_node(OP_NODE_CONTROLNET_LOADER)
class ControlnetLoaderNode(CalcNode):
    icon = "icons/in.png"
    op_code = OP_NODE_CONTROLNET_LOADER
    op_title = "ControlNet Loader"
    content_label_objname = "controlnet_loader_node"

    def __init__(self, scene):
        super().__init__(scene, inputs=[], outputs=[3])

        self.content.eval_signal.connect(self.eval)
        self.loader = ModelLoader()

    def initInnerClasses(self):
        self.content = ControlnetLoaderWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.width = 260

    def evalImplementation(self, index=0):
        self.executeChild()
        model_name = self.content.control_net_name.currentText()
        if self.value != model_name:
            self.markInvalid()
            if model_name != "":
                self.value = model_name
                self.setOutput(0, "controlnet")
                self.load_controlnet()

                self.markDirty(False)
                self.markInvalid(False)
                return self.value
            else:
                return self.value
        else:
            self.markDirty(False)
            self.markInvalid(False)
            self.grNode.setToolTip("")
            return self.value


    def load_controlnet(self):
        if not "controlnet" in gs.models:
            controlnet_path = os.path.join(self.controlnet_dir, self.control_net_name.currentText())
            gs.models["controlnet"] = load_controlnet(controlnet_path)
            return "controlnet"
        else:
            print("No reload needed")
        return "controlnet"





