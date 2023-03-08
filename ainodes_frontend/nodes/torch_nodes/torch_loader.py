import os

#from qtpy.QtWidgets import QLineEdit, QLabel, QPushButton, QFileDialog, QVBoxLayout
from qtpy import QtWidgets

from ainodes_backend.model_loader import ModelLoader
from ainodes_backend.torch_gc import torch_gc
from ainodes_frontend.nodes.base.node_config import register_node, OP_NODE_TORCH_LOADER
from ainodes_frontend.nodes.base.ai_node_base import CalcNode, CalcGraphicsNode
from ainodes_backend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_backend.node_engine.utils import dumpException
#from singleton import Singleton

#gs = Singleton()

from ainodes_backend import singleton as gs

class TorchLoaderWidget(QDMNodeContentWidget):
    def initUI(self):
        # Create a label to display the image
        # Create the dropdown widget
        self.dropdown = QtWidgets.QComboBox(self)
        self.config_dropdown = QtWidgets.QComboBox(self)
        #self.dropdown.currentIndexChanged.connect(self.on_dropdown_changed)
        # Populate the dropdown with .ckpt and .safetensors files in the checkpoints folder
        checkpoint_folder = "models/checkpoints"
        checkpoint_files = [f for f in os.listdir(checkpoint_folder) if f.endswith((".ckpt", ".safetensors"))]
        self.dropdown.addItems(checkpoint_files)
        config_folder = "models/configs"
        config_files = [f for f in os.listdir(config_folder) if f.endswith((".yaml"))]
        config_files = sorted(config_files, key=str.lower)
        self.config_dropdown.addItems(config_files)
        self.config_dropdown.setCurrentText("v1-inference_fp16.yaml")
        # Add the dropdown widget to the layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.dropdown)
        layout.addWidget(self.config_dropdown)
        self.setLayout(layout)
        self.setSizePolicy(CenterExpandingSizePolicy(self))
        self.setLayout(layout)

    def serialize(self):
        res = super().serialize()
        res["model"] = self.dropdown.currentText()
        res["config"] = self.config_dropdown.currentText()
        return res

    def deserialize(self, data, hashmap={}):
        res = super().deserialize(data, hashmap)
        try:
            self.dropdown.setCurrentText(data["model"])
            self.config_dropdown.setCurrentText(data["config"])
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
    content_label_objname = "torch_loader_node"
    category = "model"
    input_socket_name = ["EXEC"]
    output_socket_name = ["EXEC"]
    def __init__(self, scene):
        super().__init__(scene, inputs=[1], outputs=[1])
        self.loader = ModelLoader()

    def initInnerClasses(self):
        self.content = TorchLoaderWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.width = 300
        self.grNode.height = 160

    def evalImplementation(self, index=0):

        model_name = self.content.dropdown.currentText()
        config_name = self.content.config_dropdown.currentText()
        print("Loaded Model currently:", gs.loaded_models["loaded"])
        if gs.loaded_models["loaded"] != model_name:

            if model_name != "":
                if "sd" in gs.models:
                    try:
                        gs.models["sd"].cpu()
                    except:
                        pass
                    del gs.models["sd"]
                    gs.models["sd"] = None
                    torch_gc()
                self.value = model_name
                self.loader.load_model(model_name, config_name)
                self.setOutput(0, model_name)

                self.markDirty(False)
                self.markInvalid(False)
        else:
            self.markDirty(False)
            self.markInvalid(False)
            self.grNode.setToolTip("")

        if len(self.getOutputs(0)) > 0:
            self.executeChild(output_index=0)

        return self.value
    def eval(self, index=0):
        self.markDirty(True)
        self.evalImplementation(0)
    def onInputChanged(self, socket=None):
        pass



