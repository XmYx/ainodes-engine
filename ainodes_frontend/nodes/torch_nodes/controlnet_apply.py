import time

import numpy as np

import torch
#from qtpy.QtWidgets import QLineEdit, QLabel, QPushButton, QFileDialog, QVBoxLayout
from qtpy import QtWidgets, QtCore

from ainodes_backend.torch_gc import torch_gc
from ainodes_frontend.nodes.base.node_config import register_node, get_next_opcode
from ainodes_frontend.nodes.base.ai_node_base import CalcNode, CalcGraphicsNode
from ainodes_backend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_backend.node_engine.utils import dumpException
from ainodes_backend import singleton as gs
from ainodes_backend.qops import pixmap_to_pil_image

OP_NODE_CN_APPLY = get_next_opcode()
class CNApplyWidget(QDMNodeContentWidget):
    def initUI(self):
        self.strength = QtWidgets.QDoubleSpinBox()
        self.strength.setMinimum(0.01)
        self.strength.setMaximum(100.00)
        self.strength.setSingleStep(0.01)
        self.strength.setValue(1.00)

        self.button = QtWidgets.QPushButton("Run")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(15,15,15,25)
        #layout.addWidget(self.text_label)
        layout.addWidget(self.strength)
        layout.addWidget(self.button)
        self.setLayout(layout)


    def serialize(self):
        res = super().serialize()
        res['strength'] = self.strength.value()
        return res

    def deserialize(self, data, hashmap={}):
        res = super().deserialize(data, hashmap)
        try:
            self.strength.setValue(data['strength'])
            return True & res
        except Exception as e:
            dumpException(e)
        return res


@register_node(OP_NODE_CN_APPLY)
class CNApplyNode(CalcNode):
    icon = "icons/in.png"
    op_code = OP_NODE_CN_APPLY
    op_title = "Apply ControlNet"
    content_label_objname = "CN_apply_node"
    category = "controlnet"

    def __init__(self, scene):
        super().__init__(scene, inputs=[5,3,1], outputs=[3,1])


        self.content.button.clicked.connect(self.evalImplementation)
        self.busy = False
        # Create a worker object
    def initInnerClasses(self):
        self.content = CNApplyWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.height = 340
        self.grNode.width = 256
        self.content.setMinimumWidth(256)
        self.content.setMinimumHeight(256)
        self.input_socket_name = ["EXEC", "COND", "IMAGE"]
        self.output_socket_name = ["EXEC", "COND"]

        #self.content.setMinimumHeight(400)
        #self.content.setMinimumWidth(256)
        #self.content.image.changeEvent.connect(self.onInputChanged)

    def evalImplementation(self, index=0):
        self.markDirty(True)
        self.markInvalid(True)
        self.busy = False
        if self.value is None:
            """"# Start the worker thread
            self.worker = Worker(self.apply_control_net)
            # Connect the worker's finished signal to a slot that updates the node value
            self.worker.signals.result.connect(self.onWorkerFinished)
            #self.scene.queue.add_task(self.apply_control_net)
            #self.scene.queue.task_finished.connect(self.onWorkerFinished)
            self.busy = True"""
            result = self.apply_control_net()
            self.setOutput(0, result)
            self.busy = False
            # self.scene.queue.task_finished.disconnect(self.onWorkerFinished)
            if len(self.getOutputs(1)) > 0:
                self.executeChild(1)

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
    def apply_control_net(self, progress_callback=None):
        print(f"CONTROLNET APPLY NODE: Applying {gs.models['loaded_controlnet']}")
        start_time = time.time()
        try:
            cond_node, index = self.getInput(1)
            conditioning = cond_node.getOutput(index)
        except:
            conditioning = None
        try:
            latent_node, index = self.getInput(0)
            image = latent_node.getOutput(index)
        except:
            image = None

        image = pixmap_to_pil_image(image)
        image = image.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        c = []
        control_hint = image.movedim(-1,1)
        for t in conditioning:
            n = [t[0], t[1].copy()]
            c_net = gs.models["controlnet"]
            c_net.set_cond_hint(control_hint, self.content.strength.value())
            if 'control' in t[1]:
                c_net.set_previous_controlnet(t[1]['control'])
            n[1]['control'] = c_net
            c.append(n)
        #self.value = c
        self.setOutput(0, c)
        end_time = time.time()
        time_diff_ms = (end_time - start_time) * 1000
        conditioning = None
        control_hint = None
        torch_gc()
        return c
    @QtCore.Slot(object)
    def onWorkerFinished(self, result):
        # Update the node value and mark it as dirty
        #self.value = result
        self.markDirty(False)
        self.markInvalid(False)
        self.setOutput(0, result)
        self.busy = False
        #self.scene.queue.task_finished.disconnect(self.onWorkerFinished)
        if len(self.getOutputs(1)) > 0:
            self.executeChild(1)
        return
        #self.markDescendantsDirty()
        #self.evalChildren()

    def onInputChanged(self, socket=None):
        pass
        #self.eval()



















class ControlNetApply:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ),
                             "control_net": ("CONTROL_NET", ),
                             "image": ("IMAGE", ),
                             "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01})
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply_controlnet"

    CATEGORY = "conditioning"

    def apply_controlnet(self, conditioning, control_net, image, strength):
        c = []
        control_hint = image.movedim(-1,1)
        print(control_hint.shape)
        for t in conditioning:
            n = [t[0], t[1].copy()]
            c_net = control_net.copy().set_cond_hint(control_hint, strength)
            if 'control' in t[1]:
                c_net.set_previous_controlnet(t[1]['control'])
            n[1]['control'] = c_net
            c.append(n)
        return (c, )