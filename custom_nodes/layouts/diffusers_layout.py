from Qt import QtWidgets, QtCore, QtGui
from PIL.ImageQt import ImageQt

class DiffusersLayout(QtWidgets.QWidget):
    set_image_signal = QtCore.Signal(object)
    def __init__(self, parent=None, realparent=None):
        super(DiffusersLayout, self).__init__(parent)
        self.parent = realparent
        self.text_label = QtWidgets.QLabel("Diffusers:")
        self.prompt = QtWidgets.QTextEdit()
        self.steps = QtWidgets.QSpinBox()
        self.steps.setMinimum(1)
        self.steps.setMaximum(1000)
        self.steps.setValue(25)
        self.button = QtWidgets.QPushButton("Load diffusers")
        self.button.clicked.connect(self.parent.parent.load_diffusers)
        self.infer_button = QtWidgets.QPushButton("Infer Diffusers")
        self.infer_button.clicked.connect(self.parent.parent.emit_run_signal)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self.text_label)
        layout.addWidget(self.prompt)
        layout.addWidget(self.steps)
        layout.addWidget(self.button)
        layout.addWidget(self.infer_button)
    def set_value(self, value):
        self.set_image_signal.emit(value)
    @QtCore.Slot(object)
    def set_image(self, image):
        print(type(image))
        qImage = ImageQt(image)
        self.image.setPixmap(QtGui.QPixmap().fromImage(QtGui.QImage(qImage)))
