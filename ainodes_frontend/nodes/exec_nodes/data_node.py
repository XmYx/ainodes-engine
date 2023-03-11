#from qtpy.QtWidgets import QLineEdit, QLabel, QPushButton, QFileDialog, QVBoxLayout
from qtpy import QtWidgets, QtCore

from ainodes_frontend.nodes.base.node_config import register_node, get_next_opcode
from ainodes_frontend.nodes.base.ai_node_base import CalcNode, CalcGraphicsNode
from ainodes_backend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_backend.node_engine.utils import dumpException

OP_NODE_DATA = get_next_opcode()
class DataWidget(QDMNodeContentWidget):
    resize_signal = QtCore.Signal()
    def initUI(self):

        self.node_types_list = ["KSampler", "Debug"]
        self.node_data_types = {
            "KSampler":[("steps", "int"), ("scale", "float"), ("seed", "text")],
            "Debug":[("debug", "text")]
        }

        self.add_button = QtWidgets.QPushButton("Add more")
        self.print_button = QtWidgets.QPushButton("Print")
        self.print_button.clicked.connect(self.get_widget_values)

        self.node_types = QtWidgets.QComboBox()
        self.node_types.addItems(self.node_types_list)
        self.node_types.currentIndexChanged.connect(self.update_data_types)

        self.data_types = QtWidgets.QComboBox()
        self.update_data_types()

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(15,15,15,25)
        self.layout.addWidget(self.add_button)
        self.layout.addWidget(self.print_button)
        self.layout.addWidget(self.node_types)
        self.layout.addWidget(self.data_types)
        self.setLayout(self.layout)
        self.add_button.clicked.connect(self.add_widget)

    def add_widget(self):
        node_type = self.node_types.currentText()
        data_type = self.data_types.currentText()
        name = f"{node_type}_{data_type}"
        data_types = [dt for dt, _ in self.node_data_types[node_type]]
        index = data_types.index(data_type)
        _, data_type_class = self.node_data_types[node_type][index]
        widget = None
        if data_type_class == "int":
            widget = QtWidgets.QSpinBox()
        elif data_type_class == "float":
            widget = QtWidgets.QDoubleSpinBox()
        elif data_type_class == "text":
            widget = QtWidgets.QLineEdit()
        if widget is not None:
            widget.setAccessibleName(name)
            # Check if a widget with the same AccessibleName already exists
            for i in range(self.layout.count()):
                item = self.layout.itemAt(i)
                if isinstance(item, QtWidgets.QLayout):
                    for j in range(item.count()):
                        existing_widget = item.itemAt(j).widget()
                        if existing_widget and existing_widget.accessibleName() == name:
                            return
            delete_button = QtWidgets.QPushButton("Delete")
            delete_button.clicked.connect(lambda: self.layout.removeWidget(delete_button))
            delete_button.clicked.connect(lambda: self.layout.removeWidget(widget))
            delete_button.clicked.connect(widget.deleteLater)
            delete_button.clicked.connect(delete_button.deleteLater)
            hbox = QtWidgets.QHBoxLayout()
            hbox.addWidget(widget)
            hbox.addWidget(delete_button)
            self.layout.addLayout(hbox)
        self.node.resize()
    """def get_widget_values(self):
        widget_values = {}
        for i in range(self.layout.count()):
            item = self.layout.itemAt(i)
            if isinstance(item, QtWidgets.QLayout):
                for j in range(item.count()):
                    sub_item = item.itemAt(j)
                    if isinstance(sub_item, QtWidgets.QWidgetItem):
                        widget = sub_item.widget()
                        try:
                            accessible_name = widget.accessibleName()
                            print(accessible_name)
                            if accessible_name:
                                node_type, data_type = accessible_name.split("_")
                                if isinstance(widget, QtWidgets.QLineEdit):
                                    widget_values[(node_type, data_type)] = widget.text()
                                elif isinstance(widget, QtWidgets.QSpinBox):
                                    widget_values[(node_type, data_type)] = widget.value()
                                elif isinstance(widget, QtWidgets.QDoubleSpinBox):
                                    widget_values[(node_type, data_type)] = widget.value()
                        except:
                            pass
        return widget_values"""

    def update_data_types(self):
        node_type = self.node_types.currentText()
        self.data_types.clear()
        for data_type, _ in self.node_data_types[node_type]:
            self.data_types.addItem(data_type)

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


@register_node(OP_NODE_DATA)
class DataNode(CalcNode):
    icon = "icons/in.png"
    op_code = OP_NODE_DATA
    op_title = "Data"
    content_label_objname = "data_node"
    category = "debug"

    def __init__(self, scene):
        super().__init__(scene, inputs=[6,1], outputs=[6,1])
        self.interrupt = False
        self.resize()
        # Create a worker object
    def initInnerClasses(self):
        self.content = DataWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.height = 600
        self.grNode.width = 256
        self.content.setMinimumWidth(256)
        self.content.setMinimumHeight(160)
        self.input_socket_name = ["EXEC", "DATA"]
        self.output_socket_name = ["EXEC", "DATA"]


        #self.content.setMinimumHeight(400)
        #self.content.setMinimumWidth(256)
        #self.content.image.changeEvent.connect(self.onInputChanged)
    @QtCore.Slot()
    def resize(self):
        y = 300
        for i in range(self.content.layout.count()):
            item = self.content.layout.itemAt(i)
            if isinstance(item, QtWidgets.QLayout):
                for j in range(item.count()):
                    y += 15
        self.grNode.height = y + 20
        #self.grNode.width = 256
        #self.content.setMinimumWidth(256)
        self.content.setGeometry(0,0,240,y)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        self.content.setSizePolicy(size_policy)
        for socket in self.outputs + self.inputs:
            socket.setSocketPosition()
        self.updateConnectedEdges()

    def evalImplementation(self, index=0):
        self.resize()
        self.markDirty(True)
        self.markInvalid(True)

        try:
            data_node, index = self.getInput(0)
            data = data_node.getOutput(index)
        except Exception as e:
            print(e)
            data = None

        values = self.content.get_widget_values()
        print("WIDGET DATA:", values)
        #for key, value in values.items():
        #    print("DICT:", key[0], key[1], value)

        if data != None:
            data = merge_dicts(data, values)
        else:
            data = values

        print("DATA:", data)
        self.setOutput(0, data)
        if len(self.getOutputs(1)) > 0:
            self.executeChild(1)
        return None
    def onMarkedDirty(self):
        self.value = None

    def stop(self):
        self.interrupt = True
        return
    def start(self):
        self.interrupt = False
        self.evalImplementation(0)




def merge_dicts(dict1, dict2):
    result_dict = dict1.copy()
    for key, value in dict2.items():
        if key in result_dict:
            result_dict[key] = value
        else:
            result_dict[key] = value
    return result_dict





