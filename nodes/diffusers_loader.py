import torch
from PIL import Image
from PIL.ImageQt import ImageQt
from diffusers import StableDiffusionPipeline
#from qtpy.QtWidgets import QLineEdit, QLabel, QPushButton, QFileDialog, QVBoxLayout
from qtpy import QtWidgets
from qtpy.QtCore import Qt
from qtpy.QtGui import QPixmap
from nodes.base.calc_conf import register_node, OP_NODE_DIFFUSERS_LOADER
from nodes.base.calc_node_base import CalcNode, CalcGraphicsNode
from nodeeditor.node_content_widget import QDMNodeContentWidget
from nodeeditor.utils import dumpException
from singleton import Singleton

gs = Singleton()
class DiffusersLoaderWidget(QDMNodeContentWidget):
    def initUI(self):
        # Create a label to display the image
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)

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


@register_node(OP_NODE_DIFFUSERS_LOADER)
class DiffusersNode(CalcNode):
    icon = "icons/in.png"
    op_code = OP_NODE_DIFFUSERS_LOADER
    op_title = "Diffusers"
    content_label_objname = "diffusers_node"

    def __init__(self, scene):
        super().__init__(scene, inputs=[], outputs=[3])
        self.eval()
        self.content.eval_signal.connect(self.eval)

    def initInnerClasses(self):
        self.content = DiffusersLoaderWidget(self)
        self.grNode = CalcGraphicsNode(self)

        #self.content.image.changeEvent.connect(self.onInputChanged)

    def evalImplementation(self, index=0):

        if self.value is None:
            self.markInvalid()
            self.markDescendantsDirty()
            self.value = self.load_diffusers()
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
        self.markDescendantsInvalid(False)
        self.markDescendantsDirty()

        self.grNode.setToolTip("")
        self.evalChildren()

        return self.value


    def load_diffusers(self):
        if not "pipe" in gs.obj:
            repo_id = "runwayml/stable-diffusion-v1-5"
            gs.obj["pipe"] = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path=repo_id,
                                                                     torch_dtype=torch.float16,
                                                                     safety_checker=None).to("cuda")
            gs.obj["pipe"].enable_xformers_memory_efficient_attention()
            print("Diffusers model loaded")
        else:
            print("No reload needed")
        return "pipe"
