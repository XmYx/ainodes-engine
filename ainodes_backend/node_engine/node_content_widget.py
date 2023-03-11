# -*- coding: utf-8 -*-
"""A module containing the base class for the Node's content graphical representation. It also contains an example of
an overridden Text Widget, which can pass a notification to it's parent about being modified."""
from collections import OrderedDict
from ainodes_backend.node_engine.node_serializable import Serializable
from qtpy.QtWidgets import QWidget, QLabel, QVBoxLayout, QTextEdit
from qtpy import QtCore, QtWidgets
from qtpy.QtCore import Signal, QPoint, Qt


class QDMNodeContentWidget(QWidget, Serializable):
    eval_signal = QtCore.Signal(int)
    mark_dirty_signal = QtCore.Signal()

    """Base class for representation of the Node's graphics content. This class also provides layout
    for other widgets inside of a :py:class:`~node_engine.node_node.Node`"""
    def __init__(self, node:'Node', parent:QWidget=None):
        """
        :param node: reference to the :py:class:`~node_engine.node_node.Node`
        :type node: :py:class:`~nodeeditor.node_node.Node`
        :param parent: parent widget
        :type parent: QWidget

        :Instance Attributes:
            - **node** - reference to the :class:`~node_engine.node_node.Node`
            - **layout** - ``QLayout`` container
        """
        self.node = node
        super().__init__(parent)

        self.initUI()
        sshFile = "ainodes_frontend/qss/nodeeditor-dark.qss"
        with open(sshFile, "r") as fh:
            self.setStyleSheet(fh.read())


    def initUI(self):
        """Sets up layouts and widgets to be rendered in :py:class:`~node_engine.node_graphics_node.QDMGraphicsNode` class.
        """
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0,0,0,0)
        self.setLayout(self.layout)

        self.wdg_label = QLabel("Some Title")
        self.layout.addWidget(self.wdg_label)
        self.layout.addWidget(QDMTextEdit("foo"))

    def setEditingFlag(self, value:bool):
        """
        .. note::

            If you are handling keyPress events by default Qt Window's shortcuts and ``QActions``, you will not
             need to use this method.

        Helper function which sets editingFlag inside :py:class:`~node_engine.node_graphics_view.QDMGraphicsView` class.

        This is a helper function to handle keys inside nodes with ``QLineEdits`` or ``QTextEdits`` (you can
        use overridden :py:class:`QDMTextEdit` class) and with QGraphicsView class method ``keyPressEvent``.

        :param value: new value for editing flag
        """
        self.node.scene.getView().editingFlag = value

    def serialize(self) -> OrderedDict:
        return OrderedDict([
        ])

    def serializeWidgets(self, res):
        return res
        for widget in self.findChildren(QtWidgets.QWidget):
            #print(widget)
            if isinstance(widget, QtWidgets.QComboBox):
                name = widget.dynamicPropertyNames()
                print(name)
                res[name] = widget.currentText()
            elif isinstance(widget, QtWidgets.QLineEdit):
                res[widget.objectName()] = widget.text()
            elif isinstance(widget, QtWidgets.QSpinBox):
                res[widget.objectName()] = widget.value()
            elif isinstance(widget, QtWidgets.QDoubleSpinBox):
                res[widget.objectName()] = widget.value()
        print(res)
        return res

    def deserializeWidgets(self, data):
        return data
        for key, value in data.items():
            widget = self.findChild(QtWidgets.QWidget, key)
            if widget:
                if isinstance(widget, QtWidgets.QComboBox):
                    widget.setCurrentText(str(value))
                elif isinstance(widget, QtWidgets.QLineEdit):
                    widget.setText(str(value))
                elif isinstance(widget, QtWidgets.QSpinBox):
                    widget.setValue(int(value))
                elif isinstance(widget, QtWidgets.QDoubleSpinBox):
                    widget.setValue(float(value))


    def deserialize(self, data:dict, hashmap:dict={}, restore_id:bool=True) -> bool:
        return True


    def keyPressEvent(self, event):
        if event.key() == Qt.Key_E:
            print("TRIGGER")
            self.node.markDirty(True)
            self.node.eval()

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

class QDMTextEdit(QTextEdit):
    """
        .. note::

            This class is an example of a ``QTextEdit`` modification that handles the `Delete` key event with an overridden
            Qt's ``keyPressEvent`` (when not using ``QActions`` in menu or toolbar)

        Overridden ``QTextEdit`` which sends a notification about being edited to its parent's container :py:class:`QDMNodeContentWidget`
    """
    def focusInEvent(self, event:'QFocusEvent'):
        """Example of an overridden focusInEvent to mark the start of editing

        :param event: Qt's focus event
        :type event: QFocusEvent
        """
        self.parentWidget().setEditingFlag(True)
        super().focusInEvent(event)

    def focusOutEvent(self, event:'QFocusEvent'):
        """Example of an overridden focusOutEvent to mark the end of editing

        :param event: Qt's focus event
        :type event: QFocusEvent
        """
        self.parentWidget().setEditingFlag(False)
        super().focusOutEvent(event)
