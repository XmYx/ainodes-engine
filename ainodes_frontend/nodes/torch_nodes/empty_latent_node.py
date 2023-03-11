import numpy as np
import torch
from PIL import Image
#from qtpy.QtWidgets import QLineEdit, QLabel, QPushButton, QFileDialog, QVBoxLayout
from qtpy import QtWidgets, QtCore, QtGui

from ainodes_backend.resizeRight import interp_methods, resizeright
from ainodes_frontend.nodes.base.node_config import register_node, get_next_opcode
from ainodes_frontend.nodes.base.ai_node_base import CalcNode, CalcGraphicsNode
from ainodes_backend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_backend.node_engine.utils import dumpException
from ainodes_backend import singleton as gs
from ainodes_backend.qops import pixmap_to_pil_image
from einops import repeat

OP_NODE_LATENT = get_next_opcode()
OP_NODE_LATENT_COMPOSITE = get_next_opcode()
class LatentWidget(QDMNodeContentWidget):
    def initUI(self):
        # Create a label to display the image
        self.width = QtWidgets.QSpinBox()
        self.width.setMinimum(64)
        self.width.setMaximum(4096)
        self.width.setValue(512)
        self.width.setSingleStep(64)

        self.height = QtWidgets.QSpinBox()
        self.height.setMinimum(64)
        self.height.setMaximum(4096)
        self.height.setValue(512)
        self.height.setSingleStep(64)
        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor("white"))
        palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, QtGui.QColor("black"))


        self.rescale_latent = QtWidgets.QCheckBox("Latent Rescale")
        self.rescale_latent.setPalette(palette)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(15,15,15,20)
        layout.setSpacing(10)
        layout.addWidget(self.width)
        layout.addWidget(self.height)
        layout.addWidget(self.rescale_latent)

        self.setLayout(layout)

    def serialize(self):
        res = super().serialize()
        res['w'] = self.width.value()
        res['h'] = self.height.value()
        return res

    def deserialize(self, data, hashmap={}):
        res = super().deserialize(data, hashmap)
        try:
            #value = data['value']
            self.width.setValue(int(data["w"]))
            self.height.setValue(int(data["h"]))
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
        super().__init__(scene, inputs=[2,5,1], outputs=[2,1])
        #self.eval()
        #self.content.eval_signal.connect(self.eval)

    def initInnerClasses(self):
        self.content = LatentWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.input_socket_name = ["EXEC", "IMAGE", "LATENT"]
        self.output_socket_name = ["EXEC", "LATENT"]
        self.grNode.height = 210
        self.grNode.width = 200
    @QtCore.Slot(int)
    def evalImplementation(self, index=0):

        #print(self.getInput(0))
        if self.getInput(0) != None:
            #self.markInvalid()
            #self.markDescendantsDirty()

            try:
                latent_node, index = self.getInput(0)
                self.value = latent_node.getOutput(index)
                print(f"EMPTY LATENT NODE: Using Latent input with parameters: {self.value.shape}")
            except:
                print(f"EMPTY LATENT NODE: Tried using Latent input, but found an invalid value, generating latent with parameters: {self.content.width.value(), self.content.height.value()}")
                self.value = self.generate_latent()


            self.markDirty(False)
            self.markInvalid(False)
        elif self.getInput(1) != None:
            try:
                node, index = self.getInput(1)
                pixmap = node.getOutput(index)

                image = pixmap_to_pil_image(pixmap)

                image, mask_image = load_img(image,
                                             shape=(image.size[0], image.size[1]),
                                             use_alpha_as_mask=True)
                image = image.to("cuda")
                image = repeat(image, '1 ... -> b ...', b=1)

                self.value = self.encode_image(image)
                print(f"EMPTY LATENT NODE: Using Image input, encoding to Latent with parameters: {latent.shape}")
            except Exception as e:
                print(e)
        else:
            self.value = self.generate_latent()
        if self.content.rescale_latent.isChecked() == True:
            self.value = resizeright.resize(self.value, scale_factors=None,
                                         out_shape=[self.value.shape[0], self.value.shape[1], int(self.content.height.value() // 8),
                                                    int(self.content.width.value() // 8)],
                                         interp_method=interp_methods.lanczos3, support_sz=None,
                                         antialiasing=True, by_convs=True, scale_tolerance=None,
                                         max_numerator=10, pad_mode='reflect')
            print(f"Latent rescaled to: {self.value.shape}")

        self.setOutput(0, self.value)
        self.markDirty(False)
        self.markInvalid(False)
        if len(self.getOutputs(1)) > 0:
            self.executeChild(output_index=1    )
        return None
            #return self.value

    def onMarkedDirty(self):
        self.value = None
    def encode_image(self, init_image=None):
        init_latent = gs.models["sd"].model.get_first_stage_encoding(
            gs.models["sd"].model.encode_first_stage(init_image))  # move to latent space
        return init_latent

    def generate_latent(self):
        width = self.content.width.value()
        height = self.content.height.value()
        batch_size = 1
        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
        return latent

    def eval(self):
        self.markDirty(True)
        self.evalImplementation(0)
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
        super().__init__(scene, inputs=[2,2,3], outputs=[2,3])
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
def load_img(image, shape=None, use_alpha_as_mask=False):
    # use_alpha_as_mask: Read the alpha channel of the image as the mask image
    #if path.startswith('http://') or path.startswith('https://'):
    #    image = Image.open(requests.get(path, stream=True).raw)
    #else:
    #    image = Image.open(path)

    if use_alpha_as_mask:
        image = image.convert('RGBA')
    else:
        image = image.convert('RGB')

    if shape is not None:
        image = image.resize(shape, resample=Image.Resampling.LANCZOS)

    mask_image = None
    if use_alpha_as_mask:
        # Split alpha channel into a mask_image
        red, green, blue, alpha = Image.Image.split(image)
        mask_image = alpha.convert('L')
        image = image.convert('RGB')

    image = np.array(image).astype(np.float16) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    image = 2. * image - 1.

    return image, mask_image
