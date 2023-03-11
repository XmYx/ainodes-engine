import threading

#from qtpy.QtWidgets import QLineEdit, QLabel, QPushButton, QFileDialog, QVBoxLayout
from qtpy import QtWidgets, QtGui

from ainodes_frontend.nodes.base.node_config import register_node, get_next_opcode
from ainodes_frontend.nodes.base.ai_node_base import CalcNode, CalcGraphicsNode
from ainodes_backend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_backend.node_engine.utils import dumpException

OP_NODE_EXEC = get_next_opcode()

SCHEDULERS = ["karras", "normal", "simple", "ddim_uniform"]
SAMPLERS = ["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral",
            "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde",
            "dpmpp_2m", "ddim", "uni_pc", "uni_pc_bh2"]

class ExecWidget(QDMNodeContentWidget):
    def initUI(self):
        self.run_button = QtWidgets.QPushButton("Run")
        self.stop_button = QtWidgets.QPushButton("Stop")
        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor("white"))
        palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, QtGui.QColor("black"))


        self.checkbox = QtWidgets.QCheckBox("Run in thread")
        self.checkbox.setPalette(palette)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(15,15,15,25)
        layout.addWidget(self.run_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.checkbox)
        self.setLayout(layout)


    def serialize(self):
        res = super().serialize()
        return res

    def deserialize(self, data, hashmap={}):
        res = super().deserialize(data, hashmap)
        try:
            return True & res
        except Exception as e:
            dumpException(e)
        return res


@register_node(OP_NODE_EXEC)
class ExecNode(CalcNode):
    icon = "icons/in.png"
    op_code = OP_NODE_EXEC
    op_title = "Execute"
    content_label_objname = "exec_node"
    category = "debug"

    def __init__(self, scene):
        super().__init__(scene, inputs=[1], outputs=[1])
        self.content.run_button.clicked.connect(self.start)
        self.content.stop_button.clicked.connect(self.stop)

        self.interrupt = False
        # Create a worker object
    def initInnerClasses(self):
        self.content = ExecWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.height = 200
        self.grNode.width = 256
        self.content.setMinimumWidth(256)
        self.content.setMinimumHeight(160)
        self.input_socket_name = ["EXEC"]
        self.output_socket_name = ["EXEC"]

        #self.content.setMinimumHeight(400)
        #self.content.setMinimumWidth(256)
        #self.content.image.changeEvent.connect(self.onInputChanged)

    def evalImplementation(self, index=0):
        self.markDirty(True)
        self.markInvalid(True)
        if not self.interrupt:
            if len(self.getOutputs(0)) > 0:
                if self.content.checkbox.isChecked() == True:
                    thread0 = threading.Thread(target=self.executeChild, args=(0,))
                    thread0.start()
                else:
                    self.executeChild(0)
        return None
    def onMarkedDirty(self):
        self.value = None

    def stop(self):
        self.interrupt = True
        return
    def start(self):
        self.interrupt = False
        self.evalImplementation(0)










