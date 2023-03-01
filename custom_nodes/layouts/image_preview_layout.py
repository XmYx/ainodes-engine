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
        #layout.addWidget(self.image_label)
        layout.addWidget(self.image)
    def set_value(self, value):
        self.set_image_signal.emit(value)
    @QtCore.Slot(object)
    def set_image(self, image):
        # create a copy of the image data
        image_data_copy = copy.deepcopy(image)

        # convert the image data to a QPixmap and display it
        qimage = ImageQt(image_data_copy)
        pixmap = QtGui.QPixmap().fromImage(qimage)
        self.image.setPixmap(pixmap)