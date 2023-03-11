from qtpy import QtWidgets, QtCore

from ainodes_frontend.nodes.base.node_config import register_node, get_next_opcode
from ainodes_frontend.nodes.base.ai_node_base import CalcNode, CalcGraphicsNode
from ainodes_backend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_backend.node_engine.utils import dumpException

OP_NODE_CONDITIONING_COMBINE = get_next_opcode()
OP_NODE_CONDITIONING_SET_AREA = get_next_opcode()

#from singleton import Singleton
#gs = Singleton()

class ConditioningSetAreaWidget(QDMNodeContentWidget):
    def initUI(self):
        # Create a label to display the image
        #self.text_label = QtWidgets.QLabel("Diffusers:")
        self.prompt = QtWidgets.QTextEdit()

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

        self.x_spinbox = QtWidgets.QSpinBox()
        self.x_spinbox.setMinimum(0)
        self.x_spinbox.setMaximum(4096)
        self.x_spinbox.setValue(0)
        self.x_spinbox.setSingleStep(64)

        self.y_spinbox = QtWidgets.QSpinBox()
        self.y_spinbox.setMinimum(0)
        self.y_spinbox.setMaximum(4096)
        self.y_spinbox.setValue(0)
        self.y_spinbox.setSingleStep(64)

        self.strength = QtWidgets.QDoubleSpinBox()
        self.strength.setMinimum(0.01)
        self.strength.setMaximum(10.00)
        self.strength.setValue(1.00)
        self.strength.setSingleStep(0.01)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(15,15,15,20)
        layout.setSpacing(10)
        layout.addWidget(self.width)
        layout.addWidget(self.height)
        layout.addWidget(self.x_spinbox)
        layout.addWidget(self.y_spinbox)
        layout.addWidget(self.strength)
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


class ConditioningCombineWidget(QDMNodeContentWidget):
    def initUI(self):
        # Create a label to display the image
        #self.text_label = QtWidgets.QLabel("Diffusers:")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(15,15,15,20)
        layout.setSpacing(10)
        self.setLayout(layout)


    def serialize(self):
        res = super().serialize()
        #res['value'] = self.edit.text()
        return res

    def deserialize(self, data, hashmap={}):
        res = super().deserialize(data, hashmap)
        try:
            #value = data['value']
            #self.image.setPixmap(value)
            return True & res
        except Exception as e:
            dumpException(e)
        return res


@register_node(OP_NODE_CONDITIONING_COMBINE)
class ConditioningCombineNode(CalcNode):
    icon = "icons/in.png"
    op_code = OP_NODE_CONDITIONING_COMBINE
    op_title = "Combine Conditioning"
    content_label_objname = "diffusers_sampling_node"
    category = "conditioning"


    def __init__(self, scene):
        super().__init__(scene, inputs=[3,3,1], outputs=[3,1])
        #self.eval()
        #self.content.eval_signal.connect(self.evalImplementation)
        # Create a worker object
    def initInnerClasses(self):
        self.content = ConditioningCombineWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.height = 128
        self.grNode.width = 320
        #self.content.setMinimumHeight(200)
        #self.content.setMinimumWidth(320)
        self.busy = False
        #self.content.button.clicked.connect(self.exec)
        self.input_socket_name = ["EXEC", "COND", "COND2"]
        self.output_socket_name = ["EXEC", "COND"]

        #self.content.image.changeEvent.connect(self.onInputChanged)

    def evalImplementation(self, index=0):
        self.value = self.combine_conditioning()
        self.markDirty(False)
        self.markInvalid(False)
        self.setOutput(0, self.value)
        #print(self.value)
        if len(self.getOutputs(1)) > 0:
            self.executeChild(output_index=1)
        return self.value

    def onMarkedDirty(self):
        self.value = None
    def combine_conditioning(self, progress_callback=None):
        try:
            cond_1_node, index = self.getInput(1)
            conditioning1 = cond_1_node.getOutput(index)
            cond_2_node, index = self.getInput(0)
            conditioning2 = cond_2_node.getOutput(index)
            c = conditioning1 + conditioning2
            print("COND COMBINE NODE: Conditionings combined.")

            return c
        except:
            return None
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
    def onInputChanged(self, socket=None):
        pass

    def eval(self):
        self.markDirty(True)
        self.evalImplementation()

    def combine(self, conditioning_1, conditioning_2):
        return [conditioning_1 + conditioning_2]
@register_node(OP_NODE_CONDITIONING_SET_AREA)
class ConditioningAreaNode(CalcNode):
    icon = "icons/in.png"
    op_code = OP_NODE_CONDITIONING_SET_AREA
    op_title = "Set Conditioning Area"
    content_label_objname = "diffusers_sampling_node"
    category = "conditioning"

    def __init__(self, scene):
        super().__init__(scene, inputs=[3,1], outputs=[3,1])
        self.eval()
        #self.content.eval_signal.connect(self.evalImplementation)
        # Create a worker object
    def initInnerClasses(self):
        self.content = ConditioningSetAreaWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.height = 256
        self.grNode.width = 320
        self.content.setMinimumHeight(200)
        self.content.setMinimumWidth(320)
        self.busy = False
        #self.content.button.clicked.connect(self.exec)
        self.input_socket_name = ["EXEC", "COND"]
        self.output_socket_name = ["EXEC", "COND"]

        #self.content.image.changeEvent.connect(self.onInputChanged)

    def evalImplementation(self, index=0):
        try:
            cond = self.append_conditioning()
            self.setOutput(0, cond)
            print("COND AREA NODE: Conditionings Area Set.")
            self.markDirty(False)
            if len(self.getOutputs(1)) > 0:
                self.executeChild(output_index=1)
            return cond

        except:
            print("COND AREA NODE: Failed, please make sure that the conditioning is valid.")
            self.markDirty(True)
            return None
            if len(self.getOutputs(1)) > 0:
                self.executeChild(output_index=1)
            return None

    def eval(self):
        self.markDirty(True)
        self.evalImplementation()
    def onMarkedDirty(self):
        self.value = None
    def combine_conditioning(self, progress_callback=None):
        try:
            cond_1_node, index = self.getInput(1)
            conditioning1 = cond_1_node.getOutput(index)
            print("cond1", conditioning1.shape)
            cond_2_node, index = self.getInput(0)
            conditioning2 = cond_2_node.getOutput(index)
            print("cond2", conditioning2.shape)

            c = conditioning1 + conditioning2
            print(c.shape)
            return c
        except:
            return None
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
    def onInputChanged(self, socket=None):
        pass

    def exec(self):
        self.markDirty(True)
        self.markInvalid(True)
        self.value = None
        self.content.eval_signal.emit(0)

    def append_conditioning(self, progress_callback=None, min_sigma=0.0, max_sigma=99.0):
        cond_node, index = self.getInput(0)
        conditioning = cond_node.getOutput(index)
        width = self.content.width.value()
        height = self.content.height.value()
        x = self.content.x_spinbox.value()
        y = self.content.y_spinbox.value()
        strength = self.content.strength.value()
        c = []
        for t in conditioning:
            n = [t[0], t[1].copy()]
            n[1]['area'] = (height // 8, width // 8, y // 8, x // 8)
            n[1]['strength'] = strength
            n[1]['min_sigma'] = min_sigma
            n[1]['max_sigma'] = max_sigma
            c.append(n)
        return c


class ConditioningSetArea:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ),
                              "width": ("INT", {"default": 64, "min": 64, "max": 4096, "step": 64}),
                              "height": ("INT", {"default": 64, "min": 64, "max": 4096, "step": 64}),
                              "x": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 64}),
                              "y": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 64}),
                              "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "append"

    CATEGORY = "conditioning"

    def append(self, conditioning, width, height, x, y, strength, min_sigma=0.0, max_sigma=99.0):
        c = []
        for t in conditioning:
            n = [t[0], t[1].copy()]
            n[1]['area'] = (height // 8, width // 8, y // 8, x // 8)
            n[1]['strength'] = strength
            n[1]['min_sigma'] = min_sigma
            n[1]['max_sigma'] = max_sigma
            c.append(n)
        return c
