import copy

from qtpy.QtGui import QImage
from qtpy.QtCore import QRectF
from qtpy.QtWidgets import QLabel

from ainodes_backend.node_engine.node_node import Node
from ainodes_backend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_backend.node_engine.node_graphics_node import QDMGraphicsNode
from ainodes_backend.node_engine.node_node import LEFT_BOTTOM, RIGHT_BOTTOM
from ainodes_backend.node_engine.utils import dumpException
from ainodes_backend import singleton as gs


class CalcGraphicsNode(QDMGraphicsNode):
    def initSizes(self):
        super().initSizes()
        self.width = 160
        self.height = 74
        self.edge_roundness = 6
        self.edge_padding = 0
        self.title_horizontal_padding = 8
        self.title_vertical_padding = 10


    def initAssets(self):
        super().initAssets()
        self.icons = QImage("ainodes_frontend/icons/status_icons.png")

    def paint(self, painter, QStyleOptionGraphicsItem, widget=None):
        super().paint(painter, QStyleOptionGraphicsItem, widget)

        offset = 24.0
        if self.node.isDirty(): offset = 0.0
        if self.node.isInvalid(): offset = 48.0

        painter.drawImage(
            QRectF(-10, -10, 24.0, 24.0),
            self.icons,
            QRectF(offset, 0, 24.0, 24.0)
        )


class CalcContent(QDMNodeContentWidget):

    def initUI(self):
        lbl = QLabel(self.node.content_label, self)
        lbl.setObjectName(self.node.content_label_objname)



class CalcNode(Node):
    icon = ""
    op_code = 0
    op_title = "Undefined"
    content_label = ""
    content_label_objname = "calc_node_bg"
    category = "default"
    input_socket_name = ["EXEC"]
    output_socket_name = ["EXEC"]

    GraphicsNode_class = CalcGraphicsNode
    NodeContent_class = CalcContent

    def __init__(self, scene, inputs=[2,2], outputs=[1]):
        super().__init__(scene, self.__class__.op_title, inputs, outputs)

        self.value = None
        self.output_values = {}
        # it's really important to mark all nodes Dirty by default
        self.markDirty()
        self.values = {}
        #self.content.mark_dirty_signal.connect(self.markDirty)
    def getID(self, index):
        return f"{id(self)}_output_{index}"
    def setOutput(self, index, value):
        object_name = self.getID(index)
        try:
            value_copy = copy.deepcopy(value)
        except:
            try:
                value_copy = value.copy()
            except:
                value_copy = value

        gs.values[object_name] = value_copy
    def getOutput(self, index):
        object_name = self.getID(index)
        try:
            value = gs.values[object_name]
        except:
            print(f"Value doesnt exist yet, make sure to validate the node: {self.content_label_objname}")
            value = None
        return value
    def initSettings(self):
        super().initSettings()
        self.input_socket_position = LEFT_BOTTOM
        self.output_socket_position = RIGHT_BOTTOM

    def evalOperation(self, input1, input2):
        return 123

    def evalImplementation(self, index=0):
        i1 = self.getInput(0)
        i2 = self.getInput(1)

        if i1 is None or i2 is None:
            self.markInvalid()
            #self.markDescendantsDirty()
            self.grNode.setToolTip("Connect all inputs")
            return None

        else:
            val = self.evalOperation(i1.eval(), i2.eval())
            self.value = val
            self.markDirty(False)
            self.markInvalid(False)
            self.grNode.setToolTip("")

            #self.markDescendantsDirty()
            #self.evalChildren()

            return val

    def eval(self, index=0):
        if not self.isDirty() and not self.isInvalid():
            print(" _> returning cached %s value:" % self.__class__.__name__, self.value)
            return self.value
        try:
            self.evalImplementation(index)
            return None
        except ValueError as e:
            self.markInvalid()
            self.grNode.setToolTip(str(e))
            #self.markDescendantsDirty()
        except Exception as e:
            self.markInvalid()
            self.grNode.setToolTip(str(e))
            dumpException(e)


    def executeChild(self, output_index=0):
        if self.getChildrenNodes() != []:
            try:
                node = self.getOutputs(output_index)[0]
                node.markDirty(True)
                node.eval()
                return None
            except Exception as e:
                print("Skipping execution:", e)
                return None
    def onInputChanged(self, socket=None):
        print("%s::__onInputChanged" % self.__class__.__name__)
        self.markDirty(True)
        #self.content.eval_signal.emit(0)


    def serialize(self):
        res = super().serialize()
        res['op_code'] = self.__class__.op_code
        return res

    def deserialize(self, data, hashmap={}, restore_id=True):
        res = super().deserialize(data, hashmap, restore_id)
        print("Deserialized CalcNode '%s'" % self.__class__.__name__, "res:", res)
        return res
    def remove(self):
        x = 0
        for i in self.outputs:
            object_name = self.getID(x)
            gs.values[object_name] = None
            x += 1
        super().remove()
