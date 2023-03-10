import threading
import time

import numpy as np

import torch
#from qtpy.QtWidgets import QLineEdit, QLabel, QPushButton, QFileDialog, QVBoxLayout
from qtpy import QtWidgets, QtCore, QtGui

from ainodes_backend.torch_gc import torch_gc
from ainodes_backend.worker.worker import Worker
from ainodes_frontend.nodes.base.node_config import register_node, OP_NODE_DATA
from ainodes_frontend.nodes.base.ai_node_base import CalcNode, CalcGraphicsNode
from ainodes_backend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_backend.node_engine.utils import dumpException
from ainodes_backend import singleton as gs
from ainodes_frontend.nodes.qops.qimage_ops import pixmap_to_pil_image

class DataWidget(QDMNodeContentWidget):
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

    def add_widget(self):
        node_type = self.node_types.currentText()
        data_type = self.data_types.currentText()
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
            delete_button = QtWidgets.QPushButton("Delete")
            delete_button.clicked.connect(lambda: self.layout.removeWidget(delete_button))
            delete_button.clicked.connect(lambda: self.layout.removeWidget(widget))
            delete_button.clicked.connect(widget.deleteLater)
            delete_button.clicked.connect(delete_button.deleteLater)
            hbox = QtWidgets.QHBoxLayout()
            widget.setAccessibleName(f"{node_type}_{data_type}")
            hbox.addWidget(widget)
            hbox.addWidget(delete_button)
            self.layout.addLayout(hbox)

    def get_widget_values(self):
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
        return widget_values

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
        self.content.add_button.clicked.connect(self.content.add_widget)
        #self.content.stop_button.clicked.connect(self.stop)

        self.interrupt = False
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
    def resize(self):
        y = 300
        for i in range(self.content.layout.count()):
            item = self.content.layout.itemAt(i)
            if isinstance(item, QtWidgets.QLayout):
                for j in range(item.count()):
                    y += 20
        self.grNode.height = y
        #self.grNode.width = 256
        #self.content.setMinimumWidth(256)
        self.content.setMinimumHeight(y - 40)
        for socket in self.outputs + self.inputs:
            socket.setSocketPosition()
        self.updateConnectedEdges()

    def evalImplementation(self, index=0):
        self.resize()
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










