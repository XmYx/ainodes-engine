# -*- coding: utf-8 -*-
"""A module containing the base class for the Node's content graphical representation. It also contains an example of
an overridden Text Widget, which can pass a notification to it's parent about being modified."""
from collections import OrderedDict

from qtpy import QtGui

from ainodes_frontend.node_engine.node_serializable import Serializable
from qtpy.QtWidgets import QWidget, QLabel, QVBoxLayout, QTextEdit
from qtpy import QtCore, QtWidgets
from qtpy.QtCore import Qt

from ainodes_frontend.node_engine.utils import dumpException


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

    """def serialize(self) -> OrderedDict:
        return OrderedDict([
        ])"""
    def serialize(self):
        res = {}
        for i in range(self.layout().count()):
            layout = self.layout().itemAt(i)
            if isinstance(layout, QtWidgets.QHBoxLayout) or isinstance(layout, QtWidgets.QVBoxLayout):
                for k in range(layout.count()):
                    widget = layout.itemAt(k).widget()
                    if isinstance(widget, QtWidgets.QComboBox):
                        res[f"{widget.objectName()}"] = widget.currentText()
                    elif isinstance(widget, QtWidgets.QLineEdit):
                        res[f"{widget.objectName()}"] = widget.text()
                    elif isinstance(widget, QtWidgets.QTextEdit):
                        res[f"{widget.objectName()}"] = widget.toPlainText()
                    elif isinstance(widget, QtWidgets.QSpinBox) or isinstance(widget, QtWidgets.QDoubleSpinBox):
                        res[f"{widget.objectName()}"] = widget.value()
                    elif isinstance(widget, QtWidgets.QCheckBox):
                        res[f"{widget.objectName()}"] = widget.isChecked()
            elif isinstance(layout, QtWidgets.QWidgetItem):
                widget = layout.widget()
                if isinstance(widget, QtWidgets.QComboBox):
                    res[f"{widget.objectName()}"] = widget.currentText()
                elif isinstance(widget, QtWidgets.QLineEdit):
                    res[f"{widget.objectName()}"] = widget.text()
                elif isinstance(widget, QtWidgets.QTextEdit):
                    res[f"{widget.objectName()}"] = widget.toPlainText()
                elif isinstance(widget, QtWidgets.QSpinBox) or isinstance(widget, QtWidgets.QDoubleSpinBox):
                    res[f"{widget.objectName()}"] = widget.value()
                elif isinstance(widget, QtWidgets.QCheckBox):
                    res[f"{widget.objectName()}"] = widget.isChecked()
        return res
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


    #def deserialize(self, data:dict, hashmap:dict={}, restore_id:bool=True) -> bool:
    #    return True

    def deserialize(self, data, hashmap={}, restore_id:bool=True) -> bool:
        try:
            for i in range(self.layout().count()):
                layout = self.layout().itemAt(i)
                if isinstance(layout, QtWidgets.QHBoxLayout) or isinstance(layout, QtWidgets.QVBoxLayout):
                    for k in range(layout.count()):
                        widget = layout.itemAt(k).widget()
                        if isinstance(widget, QtWidgets.QComboBox):
                            index = widget.findText(data[f"{widget.objectName()}"])
                            if index == -1:
                                widget.setCurrentIndex(0)
                            else:
                                widget.setCurrentIndex(index)
                        elif isinstance(widget, QtWidgets.QLineEdit):
                            widget.setText(data[f"{widget.objectName()}"])
                        elif isinstance(widget, QTextEdit):
                            widget.setPlainText(data[f"{widget.objectName()}"])
                        elif isinstance(widget, QtWidgets.QSpinBox) or isinstance(widget, QtWidgets.QDoubleSpinBox):
                            widget.setValue(data[f"{widget.objectName()}"])
                        elif isinstance(widget, QtWidgets.QCheckBox):
                            widget.setChecked(data[f"{widget.objectName()}"])
                elif isinstance(layout, QtWidgets.QWidgetItem):
                    widget = layout.widget()
                    if isinstance(widget, QtWidgets.QComboBox):
                        index = widget.findText(data[f"{widget.objectName()}"])
                        if index == -1:
                            widget.setCurrentIndex(0)
                        else:
                            widget.setCurrentIndex(index)
                    elif isinstance(widget, QtWidgets.QLineEdit):
                        widget.setText(data[f"{widget.objectName()}"])
                    elif isinstance(widget, QTextEdit):
                        widget.setPlainText(data[f"{widget.objectName()}"])
                    elif isinstance(widget, QtWidgets.QSpinBox) or isinstance(widget, QtWidgets.QDoubleSpinBox):
                        widget.setValue(data[f"{widget.objectName()}"])
                    elif isinstance(widget, QtWidgets.QCheckBox):
                        widget.setChecked(data[f"{widget.objectName()}"])
                    return True
        except Exception as e:
            dumpException(e)
            return False
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_E:
            print("TRIGGER")
            self.node.markDirty(True)
            self.node.eval()

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
        print(widget_values)
        return widget_values"""
    def create_combo_box(self, items, label_text):
        combo_box = QtWidgets.QComboBox()
        combo_box.addItems(items)
        combo_box.setObjectName(label_text)
        label = QtWidgets.QLabel(label_text)
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(label)
        layout.addWidget(combo_box)
        combo_box.layout = layout
        return combo_box

    def create_line_edit(self, label_text):
        line_edit = QtWidgets.QLineEdit()
        label = QtWidgets.QLabel(label_text)
        label.setObjectName(label_text)
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(label)
        layout.addWidget(line_edit)
        line_edit.layout = layout
        return line_edit

    def create_spin_box(self, label_text, min_val, max_val, default_val, step_value=1):
        spin_box = QtWidgets.QSpinBox()
        spin_box.setMinimum(min_val)
        spin_box.setMaximum(max_val)
        spin_box.setValue(default_val)
        spin_box.setObjectName(label_text)
        label = QtWidgets.QLabel(label_text)
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(label)
        layout.addWidget(spin_box)
        spin_box.layout = layout
        return spin_box
    def create_double_spin_box(self, label_text, min_val, max_val, step, default_val):
        double_spin_box = QtWidgets.QDoubleSpinBox()
        double_spin_box.setMinimum(min_val)
        double_spin_box.setMaximum(max_val)
        double_spin_box.setSingleStep(step)
        double_spin_box.setValue(default_val)
        double_spin_box.setObjectName(label_text)
        label = QtWidgets.QLabel(label_text)
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(label)
        layout.addWidget(double_spin_box)
        double_spin_box.layout = layout
        return double_spin_box

    def create_check_box(self, label_text, checked=False):
        check_box = QtWidgets.QCheckBox(label_text)
        check_box.setChecked(checked)
        check_box.setObjectName(label_text)
        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor("white"))
        palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, QtGui.QColor("black"))
        check_box.setPalette(palette)
        return check_box

    def create_button_layout(self, buttons):
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(self.button)
        button_layout.addWidget(self.fix_seed_button)
        return button_layout

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
