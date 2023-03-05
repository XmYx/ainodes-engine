from PIL import Image
from PIL.ImageQt import ImageQt
from qtpy.QtWidgets import QLineEdit, QLabel, QPushButton, QFileDialog, QVBoxLayout
from qtpy.QtCore import Qt
from qtpy.QtGui import QPixmap
from nodes.base.node_config import register_node, OP_NODE_IMG_INPUT
from nodes.base.ai_node_base import CalcNode, CalcGraphicsNode
from node_engine.node_content_widget import QDMNodeContentWidget
from node_engine.utils import dumpException


class ImageInputWidget(QDMNodeContentWidget):
    def initUI(self):
        # Create a label to display the image

        self.image = QLabel(self)
        self.image.setObjectName(self.node.content_label_objname)

        ## Add a button to open the file dialog
        #self.btn = QPushButton("Select Image", self)
        #self.btn.clicked.connect(self.openFileDialog)
        fileName = self.openFileDialog()
        if fileName != None:
            image = Image.open(fileName)
            qimage = ImageQt(image)
            pixmap = QPixmap().fromImage(qimage)
            self.image.setPixmap(pixmap)
        # Create a layout to place the label and button
        layout = QVBoxLayout(self)
        layout.setContentsMargins(25, 25, 25, 25)

        layout.addWidget(self.image)
        self.setLayout(layout)

    def openFileDialog(self):
        # Open the file dialog to select a PNG file
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "PNG Files (*.png)", options=options)

        # If a file is selected, display the image in the label
        if fileName:
            return fileName
        else:
            return None
            image = Image.open(fileName)
            qimage = ImageQt(image)
            pixmap = QPixmap().fromImage(qimage)
            self.image.setPixmap(pixmap)

    def serialize(self):
        res = super().serialize()
        #res['value'] = self.edit.text()
        return res

    def deserialize(self, data, hashmap={}):
        res = super().deserialize(data, hashmap)
        try:
            value = data['value']
            self.image.setPixmap(value)
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

    def __init__(self, scene):
        super().__init__(scene, inputs=[], outputs=[3])
        self.eval()
        self.content.eval_signal.connect(self.eval)


    def initInnerClasses(self):
        self.content = ImageInputWidget(self)
        self.grNode = CalcGraphicsNode(self)


        self.content.setMinimumHeight(self.content.image.pixmap().size().height())
        self.content.setMinimumWidth(self.content.image.pixmap().size().width())
        self.grNode.height = self.content.image.pixmap().size().height() + 64
        self.grNode.width = self.content.image.pixmap().size().width() + 64

        #self.content.image.changeEvent.connect(self.onInputChanged)

    def evalImplementation(self, index=0):
        u_value = self.content.image.pixmap()
        s_value = u_value
        self.value = s_value
        self.markDirty(False)
        self.markInvalid(False)

        self.markDescendantsInvalid(False)
        self.markDescendantsDirty()

        self.grNode.setToolTip("")
        self.setOutput(0, self.content.image.pixmap())
        self.evalChildren()

        return self.content.image.pixmap()