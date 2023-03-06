import torch
#from qtpy.QtWidgets import QLineEdit, QLabel, QPushButton, QFileDialog, QVBoxLayout
from qtpy import QtWidgets, QtCore
from nodes.base.node_config import register_node, OP_NODE_LATENT,OP_NODE_LATENT_COMPOSITE
from nodes.base.ai_node_base import CalcNode, CalcGraphicsNode
from node_engine.node_content_widget import QDMNodeContentWidget
from node_engine.utils import dumpException


class LatentWidget(QDMNodeContentWidget):
    def initUI(self):
        # Create a label to display the image
        self.width = QtWidgets.QSpinBox()
        self.width.setMinimum(64)
        self.width.setMaximum(4096)
        self.width.setValue(64)
        self.width.setSingleStep(64)

        self.height = QtWidgets.QSpinBox()
        self.height.setMinimum(64)
        self.height.setMaximum(4096)
        self.height.setValue(64)
        self.height.setSingleStep(64)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(15,15,15,20)
        layout.setSpacing(10)
        layout.addWidget(self.width)
        layout.addWidget(self.height)
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


@register_node(OP_NODE_LATENT)
class LatentNode(CalcNode):
    icon = "icons/in.png"
    op_code = OP_NODE_LATENT
    op_title = "Empty Latent Image"
    content_label_objname = "diffusers_sampling_node"
    category = "latent"

    def __init__(self, scene):
        super().__init__(scene, inputs=[3], outputs=[3,3])
        self.eval()
        self.content.eval_signal.connect(self.eval)

    def initInnerClasses(self):
        self.content = LatentWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.input_socket_name = ["EXEC"]
        self.output_socket_name = ["EXEC", "LATENT"]
        self.grNode.height = 160
        self.grNode.width = 200
    @QtCore.Slot(int)
    def evalImplementation(self, index=0):

        if self.isDirty() == True:
            if self.getInput(index) != None:
                #self.markInvalid()
                #self.markDescendantsDirty()
                self.value = self.generate_latent()
                self.setOutput(0, self.value)
                self.markDirty(False)
                self.markInvalid(False)
                self.executeChild(output_index=1)
                return self.value
        else:
            self.markDirty(False)
            self.markInvalid(False)
            #self.markDescendantsDirty()
            if len(self.getOutputs(1)) > 0:
                self.executeChild(output_index=1)
            return self.value

    def onMarkedDirty(self):
        self.value = None

    def generate_latent(self):
        width = self.content.width.value()
        height = self.content.height.value()
        batch_size = 1
        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
        return latent
class LatentCompositeWidget(QDMNodeContentWidget):
    def initUI(self):
        # Create a label to display the image
        self.width = QtWidgets.QSpinBox()
        self.width.setMinimum(64)
        self.width.setMaximum(4096)
        self.width.setValue(64)
        self.width.setSingleStep(64)

        self.height = QtWidgets.QSpinBox()
        self.height.setMinimum(64)
        self.height.setMaximum(4096)
        self.height.setValue(64)
        self.height.setSingleStep(64)

        self.feather = QtWidgets.QSpinBox()
        self.feather.setMinimum(0)
        self.feather.setMaximum(200)
        self.feather.setValue(10)
        self.feather.setSingleStep(1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(15,15,15,20)
        layout.setSpacing(10)
        layout.addWidget(self.width)
        layout.addWidget(self.height)
        layout.addWidget(self.feather)
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


@register_node(OP_NODE_LATENT_COMPOSITE)
class LatentCompositeNode(CalcNode):
    icon = "icons/in.png"
    op_code = OP_NODE_LATENT_COMPOSITE
    op_title = "Composite Latent Images"
    content_label_objname = "diffusers_sampling_node"
    category = "latent"

    def __init__(self, scene):
        super().__init__(scene, inputs=[3,3,3], outputs=[3,3])
        self.eval()
        self.content.eval_signal.connect(self.eval)

    def initInnerClasses(self):
        self.content = LatentCompositeWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.input_socket_name = ["EXEC", "LATENT1", "LATENT2"]
        self.output_socket_name = ["EXEC", "LATENT"]
        self.grNode.height = 220
        self.grNode.width = 240
    @QtCore.Slot(int)
    def evalImplementation(self, index=0):

        if self.isDirty() == True:
            if self.getInput(index) != None:
                #self.markInvalid()
                #self.markDescendantsDirty()
                self.value = self.composite()
                self.setOutput(0, self.value)
                self.markDirty(False)
                self.markInvalid(False)
                if len(self.getOutputs(1)) > 0:
                    self.executeChild(output_index=1)
                return self.value
        else:
            self.markDirty(False)
            self.markInvalid(False)
            #self.markDescendantsDirty()
            if len(self.getOutputs(1)) > 0:
                self.executeChild(output_index=1)
            return self.value

    def onMarkedDirty(self):
        self.value = None

    def composite(self):
        width = self.content.width.value()
        height = self.content.height.value()
        feather = self.content.feather.value()
        x =  width // 8
        y = height // 8
        feather = feather // 8
        samples_out = self.getInput(0)
        s = self.getInput(0)
        samples_to = self.getInput(0)
        samples_from = self.getInput(1)
        if feather == 0:
            s[:,:,y:y+samples_from.shape[2],x:x+samples_from.shape[3]] = samples_from[:,:,:samples_to.shape[2] - y, :samples_to.shape[3] - x]
        else:
            samples_from = samples_from[:,:,:samples_to.shape[2] - y, :samples_to.shape[3] - x]
            mask = torch.ones_like(samples_from)
            for t in range(feather):
                if y != 0:
                    mask[:,:,t:1+t,:] *= ((1.0/feather) * (t + 1))

                if y + samples_from.shape[2] < samples_to.shape[2]:
                    mask[:,:,mask.shape[2] -1 -t: mask.shape[2]-t,:] *= ((1.0/feather) * (t + 1))
                if x != 0:
                    mask[:,:,:,t:1+t] *= ((1.0/feather) * (t + 1))
                if x + samples_from.shape[3] < samples_to.shape[3]:
                    mask[:,:,:,mask.shape[3]- 1 - t: mask.shape[3]- t] *= ((1.0/feather) * (t + 1))
            rev_mask = torch.ones_like(mask) - mask
            s[:,:,y:y+samples_from.shape[2],x:x+samples_from.shape[3]] = samples_from[:,:,:samples_to.shape[2] - y, :samples_to.shape[3] - x] * mask + s[:,:,y:y+samples_from.shape[2],x:x+samples_from.shape[3]] * rev_mask

        self.setOutput(0, s)
        #samples_out["samples"] = s
        return s
