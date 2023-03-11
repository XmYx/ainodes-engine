from PIL import Image
from PIL.ImageQt import ImageQt
from qtpy import QtCore
from qtpy.QtWidgets import QLabel, QFileDialog, QVBoxLayout
from qtpy.QtGui import QPixmap
from ainodes_frontend.nodes.base.node_config import register_node, get_next_opcode
from ainodes_frontend.nodes.base.ai_node_base import CalcNode, CalcGraphicsNode
from ainodes_backend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_backend.node_engine.utils import dumpException

OP_NODE_IMG_INPUT = get_next_opcode()

class ImageInputWidget(QDMNodeContentWidget):
    fileName = None
    parent_resize_signal = QtCore.Signal()
    def initUI(self):

        # Create a label to display the image
        self.image = QLabel(self)
        self.image.setObjectName(self.node.content_label_objname)
        self.firstRun_done = None
        # Create a layout to place the label and button
        layout = QVBoxLayout(self)
        layout.setContentsMargins(25, 25, 25, 25)
        layout.addWidget(self.image)
        self.setLayout(layout)

    def openFileDialog(self):
        # Open the file dialog to select a PNG file
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.fileName, _ = QFileDialog.getOpenFileName(None, "Select Image", "",
                                                  "PNG Files (*.png);JPEG Files (*.jpeg *.jpg);All Files(*)",
                                                  options=options)
        # If a file is selected, display the image in the label
        if self.fileName != None:
            image = Image.open(self.fileName)
            qimage = ImageQt(image)
            pixmap = QPixmap().fromImage(qimage)
            self.image.setPixmap(pixmap)
            self.parent_resize_signal.emit()

    def serialize(self):
        res = super().serialize()
        #if hasattr(self, 'fileName'):
        res['filename'] = self.fileName
        return res

    def deserialize(self, data, hashmap={}, restore_id=False):
        res = super().deserialize(data, hashmap)
        try:
            #value = data['value']
            self.fileName = data['filename']
            if self.firstRun_done == None:
                if self.fileName != None:
                    try:
                        image = Image.open(self.fileName)
                        qimage = ImageQt(image)
                        pixmap = QPixmap().fromImage(qimage)
                        self.image.setPixmap(pixmap)
                        self.firstRun_done = True
                        self.parent_resize_signal.emit()
                    except:
                        self.openFileDialog()
                elif self.fileName == None:
                    print("Opening file dialog")
                    self.openFileDialog()
            return True & res
        except Exception as e:
            dumpException(e)
        return res


@register_node(OP_NODE_IMG_INPUT)
class ImageInputNode(CalcNode):
    icon = "icons/in.png"
    op_code = OP_NODE_IMG_INPUT
    op_title = "Input"
    content_label_objname = "image_input_node"
    category = "image"
    input_socket_name = ["EXEC"]
    output_socket_name = ["EXEC", "IMAGE"]

    def __init__(self, scene):
        super().__init__(scene, inputs=[1], outputs=[5,1])
        #self.eval()
        self.content.eval_signal.connect(self.eval)
        #print(self.content.firstRun_done)
    @QtCore.Slot()
    def resize(self):
        self.content.setMinimumHeight(self.content.image.pixmap().size().height())
        self.content.setMinimumWidth(self.content.image.pixmap().size().width())
        self.grNode.height = self.content.image.pixmap().size().height() + 96
        self.grNode.width = self.content.image.pixmap().size().width() + 64

        for socket in self.outputs + self.inputs:
            socket.setSocketPosition()
        self.updateConnectedEdges()

    def init_image(self):
        #self.content.update()
        if self.content.fileName == None:
            self.content.fileName = self.openFileDialog()
        if self.content.fileName != None:
            image = Image.open(self.content.fileName)
            qimage = ImageQt(image)
            pixmap = QPixmap().fromImage(qimage)
            self.content.image.setPixmap(pixmap)
        self.resize()


    def onMarkedDirty(self):
        self.content.fileName = None
        #self.eval()

    def initInnerClasses(self):
        self.content = ImageInputWidget(self)
        self.grNode = CalcGraphicsNode(self)
        #self.init_image()
        self.content.parent_resize_signal.connect(self.resize)


        #self.content.parent_resize_signal.emit()

        #self.content.image.changeEvent.connect(self.onInputChanged)

    def evalImplementation(self, index=0):
        self.init_image()
        self.markDirty(False)
        self.markInvalid(False)
        self.grNode.setToolTip("")
        self.setOutput(0, self.content.image.pixmap())
        if len(self.getOutputs(1)) > 0:
            self.executeChild(output_index=1)
        return self.content.image.pixmap()

    def openFileDialog(self):
        # Open the file dialog to select a PNG file
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(None, "Select Image", "",
                                                  "PNG Files (*.png);JPEG Files (*.jpeg *.jpg);All Files(*)",
                                                  options=options)
        # If a file is selected, display the image in the label
        if file_name:
            return file_name
        return None
