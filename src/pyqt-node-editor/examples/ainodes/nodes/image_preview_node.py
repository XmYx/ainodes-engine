import datetime
import json
import os

from PIL.PngImagePlugin import PngInfo
# import time
#
# import PIL.Image
# import numpy as np
# import torch
# from PIL.PngImagePlugin import PngInfo
from qtpy.QtGui import QGuiApplication, QImage
from qtpy.QtWidgets import QLabel
from qtpy.QtCore import Qt
from qtpy import QtWidgets, QtGui, QtCore

# from ..ainodes_backend import pixmap_to_tensor, tensor_image_to_pixmap, tensor2pil


from examples.example_calculator.calc_node_base import CalcGraphicsNode


from examples.example_calculator.calc_conf import tensor_image_to_pixmap, tensor2pil, AiNode, register_node

from nodeeditor.node_content_widget import QDMNodeContentWidget

# from PIL import Image






class ImagePreviewWidget(QDMNodeContentWidget):
    preview_signal = QtCore.Signal(object)
    resize_signal = QtCore.Signal()
    def initUI(self):
        self.image = self.create_label("")
        # self.fps = self.create_spin_box(min_val=1, max_val=250, default_val=24, step=1, label_text="FPS")
        # self.custom_dir = self.create_line_edit("Custom save directory", placeholder="Leave empty for default")
        # # # Create checkboxes and store the horizontal layout
        #
        # self.autosave_checkbox = self.create_check_box("Autosave")
        # #self.autosave_checkbox.setChecked(True)
        #
        # self.meta_checkbox = self.create_check_box("Embed Node graph in PNG")
        # self.clipboard = self.create_check_box("Copy to Clipboard")
        #
        # checkbox_layout = self.create_horizontal_layout([
        #     self.autosave_checkbox,
        #     self.meta_checkbox,
        #     self.clipboard
        # ])
        #
        # self.button = QtWidgets.QPushButton("Save Image")
        # self.next_button = QtWidgets.QPushButton("Show Next")
        #
        # self.start_stop = QtWidgets.QPushButton("Play / Pause")
        #
        # self.widget_list.append(self.button)
        # self.widget_list.append(self.next_button)
        # self.widget_list.append(self.start_stop)

        self.create_main_layout()



@register_node()
class ImagePreviewNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/image_preview.png"
    op_title = "Image Preview"
    content_label_objname = "ImagePreviewNode"
    category = "base/image"
    output_data_ports = [0]
    NodeContent_class = ImagePreviewWidget
    dim = (600, 600)

    make_dirty = True

    def __init__(self, scene):
        super().__init__(scene, self.op_title, inputs=[2,1], outputs=[2,1])
        self.images = []
        self.pixmaps = []

        self.index = 0
        self.content.preview_signal.connect(self.show_image)
        # self.content.button.clicked.connect(self.manual_save)
        # self.content.next_button.clicked.connect(self.show_next_image)
        # self.timer = QtCore.QTimer()
        # self.timer.setInterval(40)
        # self.timer.timeout.connect(self.iter_preview)
        # self.content.start_stop.clicked.connect(self.start_stop)
        # self.content.fps.valueChanged.connect(self.set_interval)



    def evalImplementation_thread(self, index=0):
        #self.clear()
        image = self.getInputData(0)
        if image is not None:
            pixmap = tensor_image_to_pixmap(image)
            self.content.preview_signal.emit(pixmap)




    def show_image(self, image):
        self.content.image.setPixmap(image)
        self.resize(image)
    def manual_save(self):
        #for image in self.images:
        self.save_image(self.images[len(self.images) - 1][0])

    def save_image(self, pixmap, filename=None):
        try:
            image = tensor2pil(pixmap.detach().cpu().clone())
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S%f')
            os.makedirs(os.path.join("stills"), exist_ok=True)
            filename = f"stills/{timestamp}.png" if filename == None else filename

            meta_save = self.content.meta_checkbox.isChecked()

            clipboard = self.content.clipboard.isChecked()

            if meta_save:

                filename = f"stills/{timestamp}_i.png"

                metadata = PngInfo()

                json_data = self.scene.serialize()

                metadata.add_text("graph", json.dumps(json_data))

                image.save(filename, pnginfo=metadata, compress_level=4)


            else:
                image.save(filename)
            if clipboard:
                print("Copied to clipboard")
                clipboard = QGuiApplication.clipboard()
                clipboard.setImage(QImage(filename))


        except Exception as e:
            print(f"IMAGE PREVIEW NODE: Image could not be saved because: {e}")

    def resize(self, pixmap):
        # self.grNode.setToolTip("")
        # dims = [pixmap.size().height() + 360, pixmap.size().width() + 30]
        # if self.dim != dims:
        #     self.dim = dims
        #     self.grNode.height = dims[0]
        #     self.grNode.width = dims[1]

        self.grNode.width = pixmap.size().width() + 40
        self.grNode.height = pixmap.size().height() + 150

        self.content.setGeometry(15, 80, pixmap.size().width() + 30, pixmap.size().height())

        self.update_all_sockets()

    def onInputChanged(self, socket=None):
        self.markDirty(True)