import copy

from PIL.ImageQt import ImageQt
from Qt import QtWidgets, QtCore, QtGui
class ImagePreviewLayout(QtWidgets.QWidget):
    set_image_signal = QtCore.Signal(object)
    def __init__(self, parent=None):
        super(ImagePreviewLayout, self).__init__(parent)
        self.image_label = QtWidgets.QLabel("Image:")
        self.image = QtWidgets.QLabel()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self.image)
    def set_value(self, value):
        return
