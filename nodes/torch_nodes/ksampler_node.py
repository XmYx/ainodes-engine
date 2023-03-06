import secrets

import numpy as np
from einops import rearrange

from backend.k_sampler import common_ksampler

import torch
from PIL import Image
from PIL.ImageQt import ImageQt
from diffusers import StableDiffusionPipeline
#from qtpy.QtWidgets import QLineEdit, QLabel, QPushButton, QFileDialog, QVBoxLayout
from qtpy import QtWidgets, QtCore
from qtpy.QtCore import Qt
from qtpy.QtGui import QPixmap
from nodes.base.node_config import register_node, OP_NODE_K_SAMPLER
from nodes.base.ai_node_base import CalcNode, CalcGraphicsNode
from node_engine.node_content_widget import QDMNodeContentWidget
from node_engine.utils import dumpException
import singleton as gs
from worker.worker import Worker
SCHEDULERS = ["karras", "normal", "simple", "ddim_uniform"]
SAMPLERS = ["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral",
            "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde",
            "dpmpp_2m", "ddim", "uni_pc", "uni_pc_bh2"]

class KSamplerWidget(QDMNodeContentWidget):
    def initUI(self):
        # Create a label to display the image
        #self.text_label = QtWidgets.QLabel("K Sampler")
        self.schedulers = QtWidgets.QComboBox()
        self.schedulers.addItems(SCHEDULERS)

        self.sampler = QtWidgets.QComboBox()
        self.sampler.addItems(SAMPLERS)

        self.seed = QtWidgets.QLineEdit()
        self.steps = QtWidgets.QSpinBox()

        self.steps.setMinimum(1)
        self.steps.setMaximum(1000)
        self.steps.setValue(10)

        self.guidance_scale = QtWidgets.QDoubleSpinBox()
        self.guidance_scale.setMinimum(1.01)
        self.guidance_scale.setMaximum(100.00)
        self.guidance_scale.setSingleStep(0.01)
        self.guidance_scale.setValue(7.50)

        self.button = QtWidgets.QPushButton("Run")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(15,15,15,25)
        #layout.addWidget(self.text_label)
        layout.addWidget(self.schedulers)
        layout.addWidget(self.sampler)
        layout.addWidget(self.seed)
        layout.addWidget(self.steps)
        layout.addWidget(self.guidance_scale)
        layout.addWidget(self.button)
        self.setLayout(layout)


    def serialize(self):
        res = super().serialize()
        res['scheduler'] = self.schedulers.currentText()
        res['sampler'] = self.sampler.currentText()
        res['seed'] = self.seed.text()
        res['steps'] = self.steps.value()
        res['guidance_scale'] = self.guidance_scale.value()
        return res

    def deserialize(self, data, hashmap={}):
        res = super().deserialize(data, hashmap)
        try:
            self.schedulers.setCurrentText(data['scheduler'])
            self.sampler.setCurrentText(data['sampler'])
            self.seed.setText(data['seed'])
            self.steps.setValue(data['steps'])
            self.guidance_scale.setValue(data['guidance_scale'])
            return True & res
        except Exception as e:
            dumpException(e)
        return res


@register_node(OP_NODE_K_SAMPLER)
class KSamplerNode(CalcNode):
    icon = "icons/in.png"
    op_code = OP_NODE_K_SAMPLER
    op_title = "K Sampler"
    content_label_objname = "K_sampling_node"
    category = "sampling"
    def __init__(self, scene):
        super().__init__(scene, inputs=[1,1,1,1], outputs=[3,3])


        self.eval()
        self.content.eval_signal.connect(self.evalImplementation)
        self.content.button.clicked.connect(self.evalImplementation)
        self.busy = False
        # Create a worker object
    def initInnerClasses(self):
        self.content = KSamplerWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.height = 340
        self.grNode.width = 256
        self.content.setMinimumWidth(256)
        self.content.setMinimumHeight(256)
        self.input_socket_name = ["EXEC", "COND", "N COND", "LATENT"]
        self.output_socket_name = ["EXEC", "IMAGE"]

        #self.content.setMinimumHeight(400)
        #self.content.setMinimumWidth(256)
        #self.content.image.changeEvent.connect(self.onInputChanged)

    def evalImplementation(self, index=0):
        self.markDirty(True)
        #self.markInvalid(True)
        self.busy = False
        if self.value is None:
            # Start the worker thread
            #self.worker = Worker(self.k_sampling)
            # Connect the worker's finished signal to a slot that updates the node value
            #self.worker.signals.result.connect(self.onWorkerFinished)
            self.scene.queue.add_task(self.k_sampling)
            self.scene.queue.task_finished.connect(self.onWorkerFinished)
            self.busy = True
            #self.scene.threadpool.start(self.worker)
            return None
        else:
            self.markDirty(False)
            self.markInvalid(False)
            #self.markDescendantsDirty()
            #self.evalChildren()
            return self.value

    def onMarkedDirty(self):
        self.value = None
    def k_sampling(self, progress_callback=None):
        try:
            cond_node, index = self.getInput(2)
            #print("cond:", cond_node, index)
            cond = cond_node.getOutput(index)
            #print("cond value", cond)
        except:
            cond = None
        try:
            n_cond_node, index = self.getInput(1)
            n_cond = n_cond_node.getOutput(index)
        except:
            n_cond = None
        try:
            latent_node, index = self.getInput(0)
            latent = latent_node.getOutput(index)
        except:
            latent = torch.zeros([1, 4, 512 // 8, 512 // 8])
        seed = self.content.seed.text()
        try:
            seed = int(seed)
        except:
            seed = secrets.randbelow(99999999)

        sample = common_ksampler("cuda", seed, self.content.steps.value(), self.content.guidance_scale.value(), self.content.sampler.currentText(), self.content.schedulers.currentText(), cond, n_cond, latent)
        x_samples = gs.models["sd"].decode_first_stage(sample.cuda().half())
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        x_sample = 255. * rearrange(x_samples[0].cpu().numpy(), 'c h w -> h w c')
        image = Image.fromarray(x_sample.astype(np.uint8))
        qimage = ImageQt(image)
        pixmap = QPixmap().fromImage(qimage)
        self.value = pixmap
        return pixmap
    @QtCore.Slot(object)
    def onWorkerFinished(self, result):
        # Update the node value and mark it as dirty
        self.value = result
        self.markDirty(False)
        self.markInvalid(False)
        self.setOutput(0, result)
        self.busy = False
        self.scene.queue.task_finished.disconnect(self.onWorkerFinished)
        if len(self.getOutputs(1)) > 0:
            self.executeChild(output_index=1)
        return
        #self.markDescendantsDirty()
        #self.evalChildren()

    def onInputChanged(self, socket=None):
        pass
        #self.eval()

