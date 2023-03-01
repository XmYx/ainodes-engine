from Qt import QtWidgets, QtCore, QtGui
from PIL.ImageQt import ImageQt

class PromptEmbedsLayout(QtWidgets.QWidget):
    set_image_signal = QtCore.Signal(object)
    def __init__(self, parent=None, realparent=None):
        super(PromptEmbedsLayout, self).__init__(parent)
        self.parent = realparent
        self.text_label = QtWidgets.QLabel("Diffusers:")
        self.prompt = QtWidgets.QTextEdit()
        self.steps = QtWidgets.QSpinBox()
        self.steps.setMinimum(1)
        self.steps.setMaximum(1000)
        self.steps.setValue(25)
        self.button = QtWidgets.QPushButton("Load diffusers")
        self.button.clicked.connect(self.parent.parent.execute)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self.text_label)
        layout.addWidget(self.prompt)
        layout.addWidget(self.button)
    def set_value(self, value):
        return
    @QtCore.Slot(object)
    def set_image(self, image):
        return
