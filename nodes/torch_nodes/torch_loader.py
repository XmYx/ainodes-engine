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
from nodes.base.node_config import register_node, OP_NODE_TORCH_LOADER
from nodes.base.ai_node_base import CalcNode, CalcGraphicsNode
from node_engine.node_content_widget import QDMNodeContentWidget
from node_engine.utils import dumpException
#from singleton import Singleton

#gs = Singleton()

import singleton
gs = singleton
class TorchLoaderWidget(QDMNodeContentWidget):
    def initUI(self):
        # Create a label to display the image
        # Create the dropdown widget
        self.dropdown = QtWidgets.QComboBox(self)
        #self.dropdown.currentIndexChanged.connect(self.on_dropdown_changed)
        # Populate the dropdown with .ckpt and .safetensors files in the checkpoints folder
        checkpoint_folder = "models/checkpoints"
        checkpoint_files = [f for f in os.listdir(checkpoint_folder) if f.endswith((".ckpt", ".safetensors"))]
        self.dropdown.addItems(checkpoint_files)

        # Add the dropdown widget to the layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.dropdown)
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

@register_node(OP_NODE_TORCH_LOADER)
class TorchLoaderNode(CalcNode):
    icon = "icons/in.png"
    op_code = OP_NODE_TORCH_LOADER
    op_title = "Torch Loader"
    content_label_objname = "diffusers_node"

    def __init__(self, scene):
        super().__init__(scene, inputs=[], outputs=[3])

        self.content.eval_signal.connect(self.eval)
        self.loader = ModelLoader()

    def initInnerClasses(self):
        self.content = TorchLoaderWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.width = 260

    def evalImplementation(self, index=0):
        model_name = self.content.dropdown.currentText()
        if self.value != model_name:
            self.markInvalid()
            if model_name != "":
                self.value = model_name
                self.setOutput(0, model_name)
                self.loader.load_model(model_name)
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


    def load_diffusers(self, model_name):
        if not "pipe" in gs.obj:
            repo_id = model_name
            gs.obj["pipe"] = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path=repo_id,
                                                                     torch_dtype=torch.float16,
                                                                     safety_checker=None,
                                                                     use_auth_token=self.content.token.text()).to("cuda")
            gs.obj["pipe"].enable_xformers_memory_efficient_attention()
            print("Diffusers model:", model_name, "loaded")
        else:
            print("No reload needed")
        return "pipe"





