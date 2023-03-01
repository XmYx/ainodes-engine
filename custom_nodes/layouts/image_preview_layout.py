import copy

from PIL.ImageQt import ImageQt
from Qt import QtWidgets, QtCore, QtGui
class ImagePreviewLayout(QtWidgets.QWidget):
    set_image_signal = QtCore.Signal(object)
    def __init__(self, parent=None):
        super(ImagePreviewLayout, self).__init__(parent)

        self.image_label = QtWidgets.QLabel("Image:")
        self.image = QtWidgets.QLabel()

        self.set_image_signal.connect(self.set_image)
        layout = QtWidgets.QVBoxLayout(self)

        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self.image_label)
        layout.addWidget(self.image)
    def set_value(self, value):
        self.set_image_signal.emit(value)
    @QtCore.Slot(object)
    def set_image(self, image):
        #print(type(image))
        img = copy.deepcopy(image)
        qImage = ImageQt(img)
        self.image.setPixmap(QtGui.QPixmap().fromImage(QtGui.QImage(qImage)))
        #self.image.setScaledContents(True)