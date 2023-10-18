# -*- coding: utf-8 -*-
"""A module containing the base class for the Node's content graphical representation. It also contains an example of
an overridden Text Widget, which can pass a notification to it's parent about being modified."""
import re
from typing import List

from PyQt6.QtWidgets import QSpinBox, QDoubleSpinBox, QLineEdit, QComboBox, QMenu
from qtpy.QtCore import Qt
from qtpy.QtGui import QKeyEvent, QTextCursor
from qtpy import QtCore, QtWidgets
from qtpy import QtGui
from qtpy.QtWidgets import QWidget, QLabel, QVBoxLayout, QTextEdit

from ainodes_frontend.node_engine.node_serializable import Serializable


class CustomSlider(QtWidgets.QSlider):

    def __init__(self, *args, **kwargs):
        super(CustomSlider, self).__init__(*args, **kwargs)
        self.setOrientation(QtCore.Qt.Horizontal)

        # Variables to store the min and max labels
        self.min_label = QtWidgets.QLabel(str(self.minimum()))
        self.max_label = QtWidgets.QLabel(str(self.maximum()))

        # Slider layout
        self.layout = QtWidgets.QVBoxLayout(self)

        label_layout = QtWidgets.QHBoxLayout()
        # label_layout.addWidget(self.min_label)
        # label_layout.addStretch(5)
        # label_layout.addWidget(self.max_label)

        self.layout.addLayout(label_layout)
        self.layout.addWidget(self, alignment=QtCore.Qt.AlignCenter)  # Center the slider in the layout

        # Update the handle label on value change
        self.valueChanged.connect(self.update_handle_label)

    def setMinimum(self, value):
        super(CustomSlider, self).setMinimum(value)
        self.min_label.setText(str(value))

    def setMaximum(self, value):
        super(CustomSlider, self).setMaximum(value)
        self.max_label.setText(str(value))

    def update_handle_label(self, value):
        # Adjust the position of the handle label based on the current value
        handle_width = 30  # Adjust as needed
        handle_x = int(self.width() * (value - self.minimum()) / (self.maximum() - self.minimum()) - handle_width / 2)
        # Adjust the Y coordinate to position the text in the middle of the slider
        text_height = self.fontMetrics().height()
        text_width = self.fontMetrics().averageCharWidth()
        #print(len(str(value)))

        handle_label_pos = QtCore.QPoint(handle_x + text_width, int((self.height() + text_height) / 2) - 3)

        # Create a painter to draw the label on the slider
        painter = QtGui.QPainter(self)
        painter.setBrush(QtGui.QColor(70, 70, 70))  # Dark grey background
        painter.drawRect(handle_x, 0, handle_width, self.height())
        painter.setPen(QtGui.QColor(255, 255, 255))  # White text
        painter.drawText(handle_label_pos, str(value))
        painter.end()

    def paintEvent(self, event):
        # Paint the default slider
        super(CustomSlider, self).paintEvent(event)
        # Then paint the handle label
        self.update_handle_label(self.value())
class CustomSpinBox(QtWidgets.QSpinBox):
    set_signal = QtCore.Signal(int)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def contextMenuEvent(self, event):
        self.parent().contextMenuEvent(event, self)
        # propagate to parent
        #self.parent().contextMenuEvent(event)
class CustomComboBox(QtWidgets.QComboBox):
    set_signal = QtCore.Signal(int)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def contextMenuEvent(self, event):
        # propagate to parent
        self.parent().contextMenuEvent(event, self)
class CustomCheckBox(QtWidgets.QCheckBox):
    set_signal = QtCore.Signal(bool)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def contextMenuEvent(self, event):
        # propagate to parent
        self.parent().contextMenuEvent(event, self)
class CustomLineEdit(QtWidgets.QLineEdit):
    set_signal = QtCore.Signal(str)
    def __init__(self, schedule=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.schedule = schedule
    def mousePressEvent(self, event):
        if self.schedule:
            #print(self.parent().node.scene.getView().parent().window().timeline)
            self.parent().node.scene.getView().parent().window().timeline.handle_connection(self)
            #self.parent().connectWidgetToTimeline()
        super().mousePressEvent(event)
    def contextMenuEvent(self, event):
        # propagate to parent
        self.parent().contextMenuEvent(event, self)
class CustomDoubleSpinBox(QtWidgets.QDoubleSpinBox):
    set_signal = QtCore.Signal(float)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def contextMenuEvent(self, event):
        # propagate to parent
        self.parent().contextMenuEvent(event, self)


class CustomTextEdit(QtWidgets.QTextEdit):

    set_signal = QtCore.Signal(str)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.word_pattern = re.compile(r'\((.*?):([\d.]+)\)')

    def set(self, value):
        try:
            value = str(value)
        except:
            value = "Invalid Input"
        self.set_signal.emit(value)

    def get(self, html=False):
        if html:
            return self.toHtml()
        else:
            return self.toPlainText()

    def text(self):
        return self.get()

    def set_text(self, text: str = ""):
        self.setText(text)

    def mousePressEvent(self, e: QtGui.QMouseEvent) -> None:
        super().mousePressEvent(e)
        [n.doSelect(False) for n in self.parent().node.scene.nodes] # if n != self.parent().node]
        self.parent().node.doSelect(True)

    def keyPressEvent(self, event: QKeyEvent):
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier and event.key() in (Qt.Key.Key_Up, Qt.Key.Key_Down):
            self.handle_word_weighting(event.key())
        else:
            super().keyPressEvent(event)

    def handle_word_weighting(self, key):
        cursor = self.textCursor()
        if cursor.hasSelection():
            text = cursor.selectedText()
            match = self.word_pattern.fullmatch(text)
            if match:
                word = match.group(1)
                weight = float(match.group(2))
                if key == Qt.Key.Key_Up:
                    weight += 0.1
                elif key == Qt.Key.Key_Down:
                    weight -= 0.1
                weight = max(0.0, weight)  # weight should not go below 0
                new_text = f"({word}:{weight:.1f})"
            else:
                new_text = f"({text}:1.0)" if key == Qt.Key.Key_Up else f"({text}:0.9)"

            # Remember the current position of the selection
            start = cursor.selectionStart()
            end = cursor.selectionEnd()

            # Replace the text
            cursor.insertText(new_text)

            # Reselect the text
            cursor.setPosition(start, QTextCursor.MoveMode.MoveAnchor)
            cursor.setPosition(start + len(new_text), QTextCursor.MoveMode.KeepAnchor)
            self.setTextCursor(cursor)
    def contextMenuEvent(self, event):
        # propagate to parent
        self.parent().contextMenuEvent(event, self)

class QDMNodeContentWidget(QWidget, Serializable):
    eval_signal = QtCore.Signal()
    mark_dirty_signal = QtCore.Signal()
    finished = QtCore.Signal()
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

        self.widget_list = []
        self.initUI()
        for created_widget in self.widget_list:
            if isinstance(created_widget, (QSpinBox, QDoubleSpinBox)):
                created_widget.valueChanged.connect(self.mark_node_dirty)
            elif isinstance(created_widget, (QLineEdit, QTextEdit)):
                created_widget.textChanged.connect(self.mark_node_dirty)
            elif isinstance(created_widget, QComboBox):
                created_widget.currentIndexChanged.connect(self.mark_node_dirty)
    @QtCore.Slot()
    def mark_node_dirty(self, value=None):
        # print("marking")
        self.node.markDirty(True)

        #sshFile = "ainodes_frontend/qss/nodeeditor-dark.qss"
        #with open(sshFile, "r") as fh:
        #    self.setStyleSheet(fh.read())

    def initUI(self):
        """Sets up layouts and widgets to be rendered in :py:class:`~node_engine.node_graphics_node.QDMGraphicsNode` class.
        """
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0,0,0,0)
        self.setLayout(self.layout)

        self.wdg_label = QLabel("Some Title")
        self.layout.addWidget(self.wdg_label)
        self.layout.addWidget(QDMTextEdit("foo"))
    def convertWidgetToNodeInput(self, widget):
        """Converts a given widget to a node input."""
        # Step 1: Hide the widget
        widget.hide()

        # Step 2: Add a new input socket to the node
        # We will use a generic socket type for this example, you should define it appropriately.
        GENERIC_SOCKET_TYPE = 0
        self.node.initSockets(inputs=[GENERIC_SOCKET_TYPE], outputs=[], reset=False)

        # Step 3: Connect the node input to the widget's value (this is a simplistic representation and might need adjustments)
        #socket = self.node.inputs[-1]  # Get the last added input socket
        #socket.valueChanged.connect(widget.setValue)

        # Step 4: Update the node's preview
        self.node.updateConnectedEdges()
    # def contextMenuEvent(self, event, widget=None):
    #     """Override the context menu to provide an option to convert the widget to a node input."""
    #     menu = QMenu(self)
    #     convert_action = menu.addAction("Convert to Node Input")
    #     action = menu.exec(self.mapToGlobal(event.pos()))
    #
    #     if action == convert_action:
    #         #clicked_widget = self.childAt(event.pos())
    #         print(widget)
    #         if widget:  # Make sure we have a widget at the clicked position
    #             self.convertWidgetToNodeInput(widget)
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

    def serialize(self) -> dict:
        res = {}

        def serialize_widget(widget):
            if isinstance(widget, QtWidgets.QComboBox):
                res[widget.objectName()] = widget.currentText()
            elif isinstance(widget, (QtWidgets.QLineEdit, QtWidgets.QTextEdit, CustomTextEdit)):
                res[widget.objectName()] = widget.text()
            elif isinstance(widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
                res[widget.objectName()] = str(widget.value())
            elif isinstance(widget, QtWidgets.QSlider):
                res[widget.objectName()] = str(widget.value())
            elif isinstance(widget, QtWidgets.QCheckBox):
                res[widget.objectName()] = str(widget.isChecked())

        def recursive_serialize(item):
            if isinstance(item, QtWidgets.QLayout):
                for i in range(item.count()):
                    widget = item.itemAt(i).widget()
                    if widget:
                        recursive_serialize(widget)
            elif isinstance(item, QtWidgets.QWidget):
                serialize_widget(item)

        for item in self.widget_list:
            recursive_serialize(item)

        return res

    def deserialize(self, data, hashmap={}, restore_id: bool = True) -> bool:
        def deserialize_widget(widget):
            value = data.get(widget.objectName())
            if value is not None:
                if isinstance(widget, QtWidgets.QComboBox):
                    widget.setCurrentText(value)
                elif isinstance(widget, (QtWidgets.QLineEdit, QtWidgets.QTextEdit, CustomTextEdit)):
                    widget.setText(value)
                elif isinstance(widget, (QtWidgets.QSpinBox)):
                    widget.setValue(int(value))
                elif isinstance(widget, (QtWidgets.QDoubleSpinBox)):
                    widget.setValue(float(value))
                elif isinstance(widget, QtWidgets.QSlider):
                    widget.setValue(int(value))
                elif isinstance(widget, QtWidgets.QCheckBox):
                    widget.setChecked(value == "True")

        def recursive_deserialize(item):
            if isinstance(item, QtWidgets.QLayout):
                for i in range(item.count()):
                    widget = item.itemAt(i).widget()
                    if widget:
                        recursive_deserialize(widget)
            elif isinstance(item, QtWidgets.QWidget):
                deserialize_widget(item)

        for item in self.widget_list:
            recursive_deserialize(item)

        return True

    def create_combo_box(self, items, label_text, accessible_name=None, spawn=None) -> QtWidgets.QComboBox:
        """Create a combo box widget with the given items and label text.

        Args:
            items (list): List of items to be added to the combo box.
            label_text (str): Text for the label of the combo box.

        Returns:
            QtWidgets.QComboBox: A combo box widget.
        """
        combo_box = CustomComboBox()
        combo_box.addItems(items)
        combo_box.setObjectName(label_text)
        if accessible_name is not None:
            combo_box.setAccessibleName(accessible_name)
        label = QtWidgets.QLabel(label_text)
        layout = QtWidgets.QHBoxLayout()

        layout.addWidget(label)
        layout.addWidget(combo_box)
        combo_box.layout = layout
        self.widget_list.append(combo_box)
        if spawn:
            setattr(self, spawn, combo_box)
        else:
            return combo_box

    def create_line_edit(self, label_text, accessible_name=None, default=None, placeholder=None, spawn=None, schedule=None) -> QtWidgets.QLineEdit:
        """Create a line edit widget with the given label text.

        Args:
            label_text (str): Text for the label of the line edit.

        Returns:
            QtWidgets.QLineEdit: A line edit widget.
        """
        line_edit = CustomLineEdit(schedule=schedule)
        line_edit.setObjectName(label_text)
        if default is not None:
            line_edit.setText(default)
        if accessible_name is not None:
            line_edit.setAccessibleName(accessible_name)
        if placeholder is not None:
            line_edit.setPlaceholderText(placeholder)
        label = QtWidgets.QLabel(label_text)
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(label)
        layout.addWidget(line_edit)
        line_edit.layout = layout
        self.widget_list.append(line_edit)
        if spawn:
            setattr(self, spawn, line_edit)
        else:
            return line_edit
    def create_text_edit(self, label_text, placeholder="", default="", spawn=None) -> QtWidgets.QTextEdit:
        """Create a line edit widget with the given label text.

        Args:
            label_text (str): Text for the label of the line edit.

        Returns:
            QtWidgets.QLineEdit: A line edit widget.
        """
        line_edit = CustomTextEdit()
        line_edit.setText(default)
        line_edit.setPlaceholderText(placeholder)
        label = QtWidgets.QLabel(label_text)
        line_edit.setObjectName(label_text)
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(label)
        layout.addWidget(line_edit)
        line_edit.layout = layout
        line_edit.set_signal.connect(line_edit.set_text)
        self.widget_list.append(line_edit)
        if spawn:
            setattr(self, spawn, line_edit)
        else:
            return line_edit



    def create_list_view(self, label_text, spawn=None) -> QtWidgets.QListWidget:
        """Create a list widget with the given label text.

        Args:
            label_text (str): Text for the label of the line edit.

        Returns:
            QtWidgets.QListWidget: A list widget.
        """
        list_view = QtWidgets.QListWidget()
        label = QtWidgets.QLabel(label_text)
        list_view.setObjectName(label_text)
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(label)
        layout.addWidget(list_view)
        list_view.layout = layout
        self.widget_list.append(list_view)
        if spawn:
            setattr(self, spawn, list_view)
        else:
            return list_view


    def create_label(self, label_text, spawn=None) -> QtWidgets.QLabel:
        """Create a label widget with the given label text.

        Args:
            label_text (str): Text for the label of the label.

        Returns:
            QtWidgets.QLabel: A label widget.
        """
        label = QtWidgets.QLabel(label_text)
        label.setObjectName(label_text)
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(label)
        label.layout = layout
        self.widget_list.append(label)
        if spawn:
            setattr(self, spawn, label)
        else:
            return label


    def create_spin_box(self, label_text, min_val, max_val, default_val, step=1, accessible_name=None, spawn=None) -> QtWidgets.QSpinBox:
        """Create a spin box widget with the given label text, minimum value, maximum value, default value, and step value.

        Args:
            label_text (str): Text for the label of the spin box.
            min_val (int): Minimum value of the spin box.
            max_val (int): Maximum value of the spin box.
            default_val (int): Default value of the spin box.
            step (int, optional): Step value of the spin box. Defaults to 1.

        Returns:
            QtWidgets.QSpinBox: A spin box widget.
        """
        spin_box = CustomSpinBox()
        spin_box.setMinimum(min_val)
        spin_box.setMaximum(max_val)
        spin_box.setValue(default_val)
        spin_box.setSingleStep(step)
        spin_box.setObjectName(label_text)
        if accessible_name is not None:
            spin_box.setAccessibleName(accessible_name)
        label = QtWidgets.QLabel(label_text)
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(label)
        layout.addWidget(spin_box)
        spin_box.layout = layout
        self.widget_list.append(spin_box)
        if spawn:
            setattr(self, spawn, spin_box)
        else:
            return spin_box
    def create_slider(self, label_text, min_val, max_val, default_val, step=1, accessible_name=None, spawn=None) -> QtWidgets.QSlider:
        """Create a spin box widget with the given label text, minimum value, maximum value, default value, and step value.

        Args:
            label_text (str): Text for the label of the spin box.
            min_val (int): Minimum value of the spin box.
            max_val (int): Maximum value of the spin box.
            default_val (int): Default value of the spin box.
            step (int, optional): Step value of the spin box. Defaults to 1.

        Returns:
            QtWidgets.QSpinBox: A spin box widget.
        """
        slider = CustomSlider()
        slider.setOrientation(QtCore.Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(default_val)
        slider.setSingleStep(step)
        slider.setObjectName(label_text)
        if accessible_name is not None:
            slider.setAccessibleName(accessible_name)
        label = QtWidgets.QLabel(label_text)
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(label)
        layout.addWidget(slider)
        slider.layout = layout
        self.widget_list.append(slider)
        if spawn:
            setattr(self, spawn, slider)
        else:
            return slider
    def create_double_spin_box(self, label_text:str, min_val:float =0.0, max_val:float=10.0, step:float=0.01, default_val:float=1.0, accessible_name=None, spawn=None) -> QtWidgets.QDoubleSpinBox:
        """Create a double spin box widget with the given label text, minimum value, maximum value, step, and default value.

         Args:
             label_text (str): Text for the label of the double spin box.
             min_val (float): Minimum value of the double spin box.
             max_val (float): Maximum value of the double spin box.
             step (float): Step value of the double spin box.
             default_val (float): Default value of the double spin box.

         Returns:
             QtWidgets.QDoubleSpinBox: A double spin box widget.
         """
        double_spin_box = CustomDoubleSpinBox()
        double_spin_box.setMinimum(min_val)
        double_spin_box.setMaximum(max_val)
        double_spin_box.setSingleStep(step)
        double_spin_box.setValue(default_val)
        double_spin_box.setObjectName(label_text)
        if accessible_name is not None:
            double_spin_box.setAccessibleName(accessible_name)
        label = QtWidgets.QLabel(label_text)
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(label)
        layout.addWidget(double_spin_box)
        double_spin_box.layout = layout
        self.widget_list.append(double_spin_box)
        if spawn:
            setattr(self, spawn, double_spin_box)
        else:
            return double_spin_box

    def create_check_box(self, label_text, checked=False, accessible_name=None, spawn=None) -> QtWidgets.QCheckBox:
        """Create a double spin box widget with the given label text, minimum value, maximum value, step, and default value.

         Args:
             label_text (str): Text for the label of the double spin box.
             min_val (float): Minimum value of the double spin box.
             max_val (float): Maximum value of the double spin box.
             step (float): Step value of the double spin box.
             default_val (float): Default value of the double spin box.

         Returns:
             QtWidgets.QDoubleSpinBox: A double spin box widget.
         """

        check_box = CustomCheckBox(label_text)
        check_box.setChecked(checked)
        check_box.setObjectName(label_text)
        if accessible_name is not None:
            check_box.setAccessibleName(accessible_name)
        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor("white"))
        palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, QtGui.QColor("black"))
        check_box.setPalette(palette)
        self.widget_list.append(check_box)
        if spawn:
            setattr(self, spawn, check_box)
        else:
            return check_box

    def create_button_layout(self, buttons:List[QtWidgets.QPushButton], spawn=None) -> QtWidgets.QHBoxLayout:
        """Create a horizontal button layout containing the given buttons.

        Args:
            buttons (list): List of buttons to be added to the layout.

        Returns:
            QtWidgets.QHBoxLayout: A horizontal button layout.
        """
        button_layout = QtWidgets.QHBoxLayout()
        for widget in buttons:

            if isinstance(widget, QtWidgets.QCheckBox):
                palette = QtGui.QPalette()
                palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor("white"))
                palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, QtGui.QColor("black"))
                widget.setPalette(palette)
            button_layout.addWidget(widget)
        self.widget_list.append(button_layout)
        if spawn:
            setattr(self, spawn, button_layout)
        else:
            return button_layout
    def create_horizontal_layout(self, buttons, spawn=None) -> QtWidgets.QHBoxLayout:
        """Create a horizontal button layout containing the given buttons.

        Args:
            buttons (list): List of buttons to be added to the layout.

        Returns:
            QtWidgets.QHBoxLayout: A horizontal button layout.
        """
        horizontal_layout = QtWidgets.QHBoxLayout()
        for widget in buttons:

            if isinstance(widget, QtWidgets.QCheckBox):
                palette = QtGui.QPalette()
                palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor("white"))
                palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, QtGui.QColor("black"))
                widget.setPalette(palette)
            horizontal_layout.addWidget(widget)
        self.widget_list.append(horizontal_layout)
        if spawn:
            setattr(self, spawn, horizontal_layout)
        else:
            return horizontal_layout

    def create_progress_bar(self, label_text, min_val, max_val, default_val, spawn=None) -> QtWidgets.QProgressBar:
        """Create a progress bar widget with the given label text, minimum value, maximum value, and default value.

        Args:
            label_text (str): Text for the label of the progress bar.
            min_val (int): Minimum value of the progress bar.
            max_val (int): Maximum value of the progress bar.
            default_val (int): Default value of the progress bar.

        Returns:
            QtWidgets.QProgressBar: A progress bar widget.
        """
        progress_bar = QtWidgets.QProgressBar()
        progress_bar.setMinimum(min_val)
        progress_bar.setMaximum(max_val)
        progress_bar.setValue(default_val)
        progress_bar.setObjectName(label_text)
        label = QtWidgets.QLabel(label_text)
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(label)
        layout.addWidget(progress_bar)
        progress_bar.layout = layout
        self.widget_list.append(layout)
        if spawn:
            setattr(self, spawn, progress_bar)
        else:
            return progress_bar

    def create_main_layout(self, grid=None) -> None:
        """
        Create the main layout for the widget and add items from the widget_list.
        The layout is a QVBoxLayout with custom margins and will be set as the layout for the widget.
        If grid parameter is provided, a QGridLayout with grid number of columns will be created.
        """

        if self.node.use_gpu:
            from ainodes_frontend import singleton as gs
            self.create_combo_box(gs.available_gpus, "Select GPU", spawn="gpu_id")

        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setContentsMargins(15, 15, 15, 25)

        if grid:
            # Create a QGridLayout with the specified number of columns
            self.grid_layout = QtWidgets.QGridLayout()
            self.grid_layout.setSpacing(10)  # Adjust the spacing between items

            # Add widgets to the grid layout
            for i, item in enumerate(self.widget_list):
                row = i // grid
                column = i % grid
                if isinstance(item, QtWidgets.QWidget):
                    if isinstance(item, QtWidgets.QComboBox) or isinstance(item, QtWidgets.QLineEdit) or isinstance(item, QtWidgets.QSpinBox) or isinstance(item, QtWidgets.QDoubleSpinBox) or isinstance(item, QtWidgets.QSlider):
                        self.grid_layout.addLayout(item.layout, row, column)
                    else:
                        self.grid_layout.addWidget(item, row, column)
                elif isinstance(item, QtWidgets.QLayout):
                    self.grid_layout.addLayout(item, row, column)

            self.main_layout.addLayout(self.grid_layout)
        else:
            # Add items to the main layout without a grid
            for item in self.widget_list:
                if isinstance(item, QtWidgets.QLayout):
                    self.main_layout.addLayout(item)
                elif isinstance(item, QtWidgets.QWidget):
                    self.main_layout.addWidget(item)
        #self.deafult_run_button = QtWidgets.QPushButton("Evaluate Node")
        #self.main_layout.addWidget(self.deafult_run_button)
        #self.deafult_run_button.clicked.connect(self.node.evalImplementation)
        self.setLayout(self.main_layout)

    def wheelEvent(self, event: QtGui.QWheelEvent):
        pass
        #event.ignore()
        #print("IGNORE IN CONTENT WIDGET")


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
