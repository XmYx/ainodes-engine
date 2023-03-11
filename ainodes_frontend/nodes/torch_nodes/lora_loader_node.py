import os
from qtpy import QtWidgets

from ainodes_backend.lora_loader import load_lora_for_models
from ainodes_backend.model_loader import ModelLoader
from ainodes_backend.torch_gc import torch_gc
from ainodes_frontend.nodes.base.node_config import register_node, get_next_opcode
from ainodes_frontend.nodes.base.ai_node_base import CalcNode, CalcGraphicsNode
from ainodes_backend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_backend.node_engine.utils import dumpException
from ainodes_backend import singleton as gs

OP_NODE_LORA_LOADER = get_next_opcode()
class LoraLoaderWidget(QDMNodeContentWidget):
    def initUI(self):
        # Create a label to display the image
        # Create the dropdown widget
        self.dropdown = QtWidgets.QComboBox(self)
        # Populate the dropdown with .ckpt and .safetensors files in the checkpoints folder
        lora_folder = "models/loras"
        lora_files = [f for f in os.listdir(lora_folder) if f.endswith(('.safetensors', '.ckpt', '.pt', '.bin', '.pth'))]
        if lora_files == []:
            self.dropdown.addItem("Please place a lora in models/loras")
            print(f"LORA LOADER NODE: No model file found at {os.getcwd()}/models/loras,")
            print(f"LORA LOADER NODE: please download your favorite ckpt before Evaluating this node.")
        self.dropdown.addItems(lora_files)
        # Add the dropdown widget to the layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.dropdown)
        self.setLayout(layout)
        self.setSizePolicy(CenterExpandingSizePolicy(self))
        self.setLayout(layout)

    def serialize(self):
        res = super().serialize()
        res["model"] = self.dropdown.currentText()
        return res

    def deserialize(self, data, hashmap={}):
        res = super().deserialize(data, hashmap)
        try:
            self.dropdown.setCurrentText(data["model"])
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

@register_node(OP_NODE_LORA_LOADER)
class TorchLoaderNode(CalcNode):
    icon = "icons/in.png"
    op_code = OP_NODE_LORA_LOADER
    op_title = "Lora Loader"
    content_label_objname = "lora_loader_node"
    category = "model"
    input_socket_name = ["EXEC"]
    output_socket_name = ["EXEC"]
    def __init__(self, scene):
        super().__init__(scene, inputs=[1], outputs=[1])
        #self.loader = ModelLoader()

    def initInnerClasses(self):
        self.content = LoraLoaderWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.width = 340
        self.grNode.height = 160

    def evalImplementation(self, index=0):
        file = self.content.dropdown.currentText()

        self.load_lora_to_ckpt(file)
        if len(self.getOutputs(0)) > 0:
            self.executeChild(output_index=0)

        return self.value
    def eval(self, index=0):
        self.markDirty(True)
        self.evalImplementation(0)
    def onInputChanged(self, socket=None):
        pass

    def load_lora_to_ckpt(self, lora_name):
        lora_path = os.path.join("models/loras", lora_name)
        strength_model = 1.0
        strength_clip = 1.0
        load_lora_for_models(lora_path, strength_model, strength_clip)



