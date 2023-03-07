import torch
from diffusers import StableDiffusionPipeline
#from qtpy.QtWidgets import QLineEdit, QLabel, QPushButton, QFileDialog, QVBoxLayout
from qtpy import QtWidgets
from ainodes_frontend.nodes.base.node_config import register_node, OP_NODE_DIFFUSERS_LOADER
from ainodes_frontend.nodes.base.ai_node_base import CalcNode, CalcGraphicsNode
from ainodes_backend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_backend.node_engine.utils import dumpException
import ainodes_backend.singleton as gs

class DiffusersLoaderWidget(QDMNodeContentWidget):
    def initUI(self):
        # Create a label to display the image
        self.model_label = QtWidgets.QLabel("Model")
        self.model_name = QtWidgets.QLineEdit()
        self.token_label = QtWidgets.QLabel("Token")
        self.token = QtWidgets.QLineEdit()
        self.load_button = QtWidgets.QPushButton("Load")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self.model_label)
        layout.addWidget(self.model_name)
        layout.addWidget(self.token_label)
        layout.addWidget(self.token)
        layout.addWidget(self.load_button)
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

@register_node(OP_NODE_DIFFUSERS_LOADER)
class DiffusersNode(CalcNode):
    icon = "icons/in.png"
    op_code = OP_NODE_DIFFUSERS_LOADER
    op_title = "Diffusers"
    content_label_objname = "diffusers_node"
    category = "model"


    def __init__(self, scene):
        super().__init__(scene, inputs=[], outputs=[3])
        self.eval()
        self.content.eval_signal.connect(self.eval)
        self.content.load_button.clicked.connect(self.eval)

    def initInnerClasses(self):
        self.content = DiffusersLoaderWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.height = 400
        self.grNode.width = 320
        #self.content.setMinimumHeight(390)
        #self.content.setMinimumWidth(300)

        #self.content.image.changeEvent.connect(self.onInputChanged)

    def evalImplementation(self, index=0):
        model_name = self.content.model_name.text()
        if self.value != model_name:
            self.markInvalid()
            #self.markDescendantsDirty()
            if model_name != "":
                self.value = self.load_diffusers(model_name)
                return self.value
            else:
                return self.value
        else:
            self.markDirty(False)
            self.markInvalid(False)
            self.grNode.setToolTip("")

            #self.markDescendantsDirty()
            #self.evalChildren()

            return self.value


        #u_value = self.content.image.pixmap()

        self.markDirty(False)
        self.markInvalid(False)
        #self.markDescendantsInvalid(False)
        #self.markDescendantsDirty()
        self.grNode.setToolTip("")
        #self.evalChildren()
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
