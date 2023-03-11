import secrets
import threading

import numpy as np
from einops import rearrange

from ainodes_backend.inpaint.generator import run_inpaint
from ainodes_backend.k_sampler import common_ksampler

import torch
from PIL import Image
from PIL.ImageQt import ImageQt
#from qtpy.QtWidgets import QLineEdit, QLabel, QPushButton, QFileDialog, QVBoxLayout
from qtpy import QtWidgets, QtCore
from qtpy.QtGui import QPixmap

from ainodes_backend.torch_gc import torch_gc
from ainodes_frontend.nodes.base.node_config import register_node, get_next_opcode
from ainodes_frontend.nodes.base.ai_node_base import CalcNode, CalcGraphicsNode
from ainodes_backend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_backend.node_engine.utils import dumpException
from ainodes_backend import singleton as gs
from ainodes_backend.qops import pixmap_to_pil_image, pil_image_to_pixmap

OP_NODE_INPAINT = get_next_opcode()
class InpaintWidget(QDMNodeContentWidget):
    def initUI(self):
        # Create a label to display the image
        #self.text_label = QtWidgets.QLabel("K Sampler")


        self.seed_layout = QtWidgets.QHBoxLayout()
        self.seed_label = QtWidgets.QLabel("Seed:")
        self.seed = QtWidgets.QLineEdit()
        self.seed_layout.addWidget(self.seed_label)
        self.seed_layout.addWidget(self.seed)

        self.prompt = QtWidgets.QTextEdit()

        self.steps_layout = QtWidgets.QHBoxLayout()
        self.steps_label = QtWidgets.QLabel("Steps:")
        self.steps = QtWidgets.QSpinBox()
        self.steps.setMinimum(1)
        self.steps.setMaximum(1000)
        self.steps.setValue(10)
        self.steps_layout.addWidget(self.steps_label)
        self.steps_layout.addWidget(self.steps)


        self.guidance_scale_layout = QtWidgets.QHBoxLayout()
        self.guidance_scale_label = QtWidgets.QLabel("Guidance Scale:")
        self.guidance_scale = QtWidgets.QDoubleSpinBox()
        self.guidance_scale.setMinimum(1.01)
        self.guidance_scale.setMaximum(100.00)
        self.guidance_scale.setSingleStep(0.01)
        self.guidance_scale.setValue(7.50)
        self.guidance_scale_layout.addWidget(self.guidance_scale_label)
        self.guidance_scale_layout.addWidget(self.guidance_scale)

        self.button_layout = QtWidgets.QHBoxLayout()
        self.button = QtWidgets.QPushButton("Run")
        self.fix_seed_button = QtWidgets.QPushButton("Fix Seed")
        self.button_layout.addWidget(self.button)
        self.button_layout.addWidget(self.fix_seed_button)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(15,15,15,25)
        layout.addLayout(self.seed_layout)
        layout.addWidget(self.prompt)
        layout.addLayout(self.steps_layout)
        layout.addLayout(self.guidance_scale_layout)
        layout.addLayout(self.button_layout)

        self.setLayout(layout)


    def serialize(self):
        res = super().serialize()
        res['seed'] = self.seed.text()
        res['steps'] = self.steps.value()
        res['guidance_scale'] = self.guidance_scale.value()
        return res

    def deserialize(self, data, hashmap={}):
        res = super().deserialize(data, hashmap)
        try:
            self.seed.setText(data['seed'])
            self.steps.setValue(data['steps'])
            self.guidance_scale.setValue(data['guidance_scale'])
            return True & res
        except Exception as e:
            dumpException(e)
        return res


@register_node(OP_NODE_INPAINT)
class InpaintNode(CalcNode):
    icon = "icons/in.png"
    op_code = OP_NODE_INPAINT
    op_title = "InPaint Alpha"
    content_label_objname = "K_sampling_node"
    category = "sampling"
    def __init__(self, scene):
        super().__init__(scene, inputs=[5,5,1], outputs=[5,1])
        self.content.button.clicked.connect(self.evalImplementation)
        self.busy = False
        # Create a worker object
    def initInnerClasses(self):
        self.content = InpaintWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.height = 500
        self.grNode.width = 256
        self.content.setMinimumWidth(256)
        self.content.setMinimumHeight(256)
        self.input_socket_name = ["EXEC", "IMAGE", "IMAGE"]
        self.output_socket_name = ["EXEC", "IMAGE"]
        self.seed = ""
        self.content.fix_seed_button.clicked.connect(self.setSeed)
        #self.content.setMinimumHeight(400)
        #self.content.setMinimumWidth(256)
        #self.content.image.changeEvent.connect(self.onInputChanged)

    def evalImplementation(self, index=0):


        self.markDirty(True)
        #self.markInvalid(True)
        #self.busy = False
        #self.worker = Worker(self.k_sampling)
        # Connect the worker's finished signal to a slot that updates the node value
        #self.worker.signals.result.connect(self.onWorkerFinished)
        #self.scene.queue.add_task(self.k_sampling)
        #self.scene.queue.task_finished.connect(self.onWorkerFinished)
        self.busy = True
        #self.scene.threadpool.start(self.worker)
        thread0 = threading.Thread(target=self.inpainting)
        thread0.start()


        return None

    def onMarkedDirty(self):
        self.value = None

    def inpainting(self):
        try:
            image_input_node, index = self.getInput(1)
            image_pixmap = image_input_node.getOutput(index)
        except Exception as e:
            print(e)
        try:
            mask_input_node, index = self.getInput(0)
            mask_pixmap = mask_input_node.getOutput(index)
        except Exception as e:
            print(e)

        init_image = pixmap_to_pil_image(image_pixmap)
        mask_image = pixmap_to_pil_image(mask_pixmap)

        prompt = self.content.prompt.toPlainText()
        try:
            seed = self.content.seed.text()
            seed = int(seed)
        except:
            seed = secrets.randbelow(99999999)
        scale = self.content.guidance_scale.value()
        steps = self.content.steps.value()
        blend_mask = 5
        mask_blur = 5
        recons_blur = 5

        result = run_inpaint(init_image, mask_image, prompt, seed, scale, steps, blend_mask, mask_blur, recons_blur)
        pixmap = pil_image_to_pixmap(result)
        self.setOutput(0, pixmap)
        if len(self.getOutputs(1)) > 0:
            self.executeChild(output_index=1)


    def k_sampling(self, progress_callback=None):
        try:
            cond_node, index = self.getInput(2)
            cond = cond_node.getOutput(index)
        except Exception as e:
            print(e)
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
        self.seed = self.content.seed.text()
        try:
            self.seed = int(self.seed)
        except:
            self.seed = secrets.randbelow(99999999)
        try:
            last_step = self.content.steps.value() if self.content.stop_early.isChecked() == False else self.content.last_step.value()
            sample = common_ksampler(device="cuda",
                                     seed=self.seed,
                                     steps=self.content.steps.value(),
                                     start_step=self.content.start_step.value(),
                                     last_step=last_step,
                                     cfg=self.content.guidance_scale.value(),
                                     sampler_name=self.content.sampler.currentText(),
                                     scheduler=self.content.schedulers.currentText(),
                                     positive=cond,
                                     negative=n_cond,
                                     latent=latent,
                                     disable_noise=self.content.disable_noise.isChecked(),
                                     force_full_denoise=self.content.force_denoise.isChecked(),
                                     denoise=self.content.denoise.value())

            return_sample = sample.cpu().half()

            x_samples = gs.models["sd"].decode_first_stage(sample.half())
            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
            x_sample = 255. * rearrange(x_samples[0].cpu().numpy(), 'c h w -> h w c')
            image = Image.fromarray(x_sample.astype(np.uint8))
            qimage = ImageQt(image)
            pixmap = QPixmap().fromImage(qimage)
            self.value = pixmap
            del sample
            del x_samples
            x_samples = None
            sample = None
            torch_gc()
            self.onWorkerFinished([pixmap, return_sample])
        except:
            self.busy = False
            if len(self.getOutputs(2)) > 0:
                self.executeChild(output_index=2)
        return [pixmap, return_sample]
    @QtCore.Slot(object)
    def onWorkerFinished(self, result):
        # Update the node value and mark it as dirty
        #self.value = result[0]
        print("K SAMPLER:", self.content.steps.value(), "steps,", self.content.sampler.currentText(), " seed: ", self.seed)
        self.markDirty(False)
        self.markInvalid(False)
        self.setOutput(0, result[0])
        self.setOutput(1, result[1])
        self.busy = False
        #self.worker.autoDelete()
        #self.scene.queue.task_finished.disconnect(self.onWorkerFinished)
        if len(self.getOutputs(2)) > 0:
            self.executeChild(output_index=2)
        return
        #self.markDescendantsDirty()
        #self.evalChildren()
    def setSeed(self):
        self.content.seed.setText(str(self.seed))
    def onInputChanged(self, socket=None):
        pass
        #self.eval()

