from qtpy import QtWidgets, QtCore, QtGui

from nodes.base.node_config import register_node, OP_NODE_CONDITIONING
from nodes.base.ai_node_base import CalcNode, CalcGraphicsNode
from node_engine.node_content_widget import QDMNodeContentWidget
from node_engine.utils import dumpException

from worker.worker import Worker
#from singleton import Singleton
#gs = Singleton()

import singleton as gs
class ConditioningWidget(QDMNodeContentWidget):
    def initUI(self):
        # Create a label to display the image
        #self.text_label = QtWidgets.QLabel("Diffusers:")
        self.prompt = QtWidgets.QTextEdit()
        self.steps = QtWidgets.QSpinBox()
        self.steps.setMinimum(1)
        self.steps.setMaximum(1000)
        self.steps.setValue(25)
        self.button = QtWidgets.QPushButton("Get Conditioning")
        #self.button.clicked.connect(self.parent.parent.eval)
        #self.infer_button = QtWidgets.QPushButton("")
        #self.infer_button.clicked.connect(self.parent.parent.emit_run_signal)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(15,15,15,20)
        layout.setSpacing(10)
        #layout.addWidget(self.text_label)
        layout.addWidget(self.prompt)
        layout.addWidget(self.steps)
        layout.addWidget(self.button)
        #layout.addWidget(self.infer_button)
        self.setLayout(layout)


    def serialize(self):
        res = super().serialize()
        #res['value'] = self.edit.text()
        return res

    def deserialize(self, data, hashmap={}):
        res = super().deserialize(data, hashmap)
        try:
            value = data['value']
            self.image.setPixmap(value)
            return True & res
        except Exception as e:
            dumpException(e)
        return res


@register_node(OP_NODE_CONDITIONING)
class ConditioningNode(CalcNode):
    icon = "icons/in.png"
    op_code = OP_NODE_CONDITIONING
    op_title = "Conditioning"
    content_label_objname = "diffusers_sampling_node"

    def __init__(self, scene):
        super().__init__(scene, inputs=[3], outputs=[4,4])
        self.eval()
        self.content.eval_signal.connect(self.evalImplementation)
        # Create a worker object
    def initInnerClasses(self):
        self.content = ConditioningWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.height = 256
        self.grNode.width = 320
        self.content.setMinimumHeight(200)
        self.content.setMinimumWidth(320)
        self.busy = False
        self.content.button.clicked.connect(self.exec)
        self.input_socket_name = ["EXEC"]
        self.output_socket_name = ["EXEC", "COND"]

        #self.content.image.changeEvent.connect(self.onInputChanged)

    def evalImplementation(self, index=0):
        if self.value is None:
            # Start the worker thread
            #self.worker = Worker(self.get_conditioning)
            # Connect the worker's finished signal to a slot that updates the node value
            #self.worker.signals.result.connect(self.onWorkerFinished)
            self.scene.queue.add_task(self.get_conditioning)
            self.scene.queue.task_finished.connect(self.onWorkerFinished)
            self.busy = True
            #self.scene.threadpool.start(self.worker)
            return None
        else:
            self.markDirty(False)
            self.markInvalid(False)
            #self.markDescendantsDirty()
            #self.evalChildren()
            self.executeChild()
            return self.value

    def onMarkedDirty(self):
        self.value = None
    def get_conditioning(self, progress_callback=None):
        #print("Getting Conditioning on ", id(self))
        prompt = self.content.prompt.toPlainText()
        c = gs.models["sd"].cond_stage_model.encode([prompt])
        uc = {}
        return [[c, uc]]
    @QtCore.Slot(object)
    def onWorkerFinished(self, result):
        # Update the node value and mark it as dirty
        self.value = result
        self.scene.queue.task_finished.disconnect(self.onWorkerFinished)
        self.setOutput(0, result)
        self.markDirty(False)
        self.markInvalid(False)
        self.busy = False
        if len(self.getOutputs(1)) > 0:
            self.executeChild(output_index=1)
        return
        #self.markDescendantsDirty()
        #self.evalChildren()
    def onInputChanged(self, socket=None):
        pass

    def exec(self):
        self.markDirty(True)
        self.markInvalid(True)
        self.value = None
        self.content.eval_signal.emit(0)

SCHEDULERS = ["karras", "normal", "simple", "ddim_uniform"]
SAMPLERS = ["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral",
            "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde",
            "dpmpp_2m", "ddim", "uni_pc", "uni_pc_bh2"]
