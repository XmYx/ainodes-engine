from NodeGraphQt import NodeBaseWidget
from PIL.ImageQt import ImageQt
from Qt import QtCore, QtGui, QtWidgets


class ImagePreviewWidget(NodeBaseWidget):

    set_image_signal = QtCore.Signal(object)

    def __init__(self, node, parent=None):
        super(ImagePreviewWidget, self).__init__(node, parent)

        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.layout().addWidget(self.image_label)

    @QtCore.Slot(object)
    def set_image(self, image_data):
        # create a copy of the image data
        image_data_copy = image_data.copy()

        # convert the image data to a QPixmap and display it
        qimage = ImageQt(image_data_copy)
        pixmap = QtGui.QPixmap().fromImage(qimage)
        self.image_label.setPixmap(pixmap)


class ImagePreviewNode(BaseNode):

    # Unique node identifier.
    __identifier__ = 'com.example.nodes.ImagePreviewNode'

    # Initial default node name.
    NODE_NAME = 'Image Preview'

    def __init__(self):
        super(ImagePreviewNode, self).__init__()

        # create input port
        self.add_input('Image')

        # create custom widget to display image
        self.image_widget = ImagePreviewWidget(self)
        self.image_widget.set_image_signal.connect(self.image_widget.set_image)
        self.set_widget(self.image_widget)

    def process(self):
        # get the input image data
        image_data = self.get_input_data(0)

        # set the output to the input data
        self.set_output_data(0, image_data)

        # emit signal to update image widget
        self.image_widget.set_image_signal.emit(image_data)
