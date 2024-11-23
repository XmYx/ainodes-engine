import io
import os
import numpy as np
import torch
from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import pyqtSignal, QUrl
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QFileDialog, QLabel, QVBoxLayout

from node_core.node_register import tensor_image_to_pixmap, AiNode, register_node
from nodeeditor.node_content_widget import QDMNodeContentWidget



class ImagePreviewWidget(QDMNodeContentWidget):
    set_image_signal = pyqtSignal(QPixmap)

    def initUI(self):
        # Create label to display the image
        self.image_label = QLabel()
        self.image_label.setScaledContents(False)  # Don't scale the image
        self.image_label.setPixmap(QPixmap())  # Start with an empty pixmap

        # Create a checkbox for saving the output
        self.create_check_box('Save Output', checked=False, spawn='save_output', object_name='save_output')

        # Set up the layout
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.save_output)
        self.setLayout(layout)

        # Connect the signal
        self.set_image_signal.connect(self.image_label.setPixmap)

from node_core.node_register import register_node, AiNode

@register_node()
class ImagePreviewNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/input_image.png"
    op_code = 0
    op_title = "Image Preview"
    content_label_objname = "image_preview_node"
    category = "base/image"
    custom_input_socket_names = ["IMAGE", "EXEC"]
    dim = (300, 120)
    NodeContent_class = ImagePreviewWidget

    def __init__(self, scene):
        super().__init__(scene, inputs=[5,1], outputs=[1])

    def evalImplementation_thread(self, index=0):
        image_input = self.getInputData(0)
        if image_input is not None:
            # Handle different types of input
            if isinstance(image_input, dict):
                # If image_input is a dict, try to get 'samples'
                image_tensor = image_input.get('samples', None)
            else:
                image_tensor = image_input
            if image_tensor is not None:
                # Assuming image_tensor is a batched tensor of shape [batch_size, C, H, W] or [batch_size, H, W, C]
                if len(image_tensor.shape) == 4:
                    # Batch dimension present
                    image_tensor = image_tensor[0]  # Take first image in batch
                # Now image_tensor should be of shape [C, H, W] or [H, W, C]
                if image_tensor.shape[0] == 3 or image_tensor.shape[0] == 4:
                    # Shape is [C, H, W], transpose to [H, W, C]
                    image_np = image_tensor.cpu().numpy()
                    image_np = np.transpose(image_np, (1, 2, 0))
                else:
                    # Shape is [H, W, C]
                    image_np = image_tensor.cpu().numpy()
                # Convert from float [0,1] to uint8 [0,255]
                image_np = (image_np * 255).clip(0, 255).astype(np.uint8)
                # Handle possible alpha channel
                if image_np.shape[2] == 4:
                    # Has alpha channel
                    image_pil = Image.fromarray(image_np, mode='RGBA')
                else:
                    image_pil = Image.fromarray(image_np)
                # Convert to QPixmap
                pixmap = QPixmap.fromImage(ImageQt(image_pil))
                self.resize(pixmap)
                self.content.set_image_signal.emit(pixmap)
                # If save output is checked, save the image
                if self.content.save_output.isChecked():
                    # Save the image
                    output_path = 'output_image.png'  # Customize the path or filename as needed
                    image_pil.save(output_path)
                    print(f"Image saved to {output_path}")

    def resize(self, pixmap):
        self.grNode.setToolTip("")
        dims = [pixmap.size().height() + 360, pixmap.size().width() + 30]
        if self.dim != dims:
            self.dim = dims
            self.grNode.height = dims[0]
            self.grNode.width = dims[1]
            self.content.setGeometry(0, 25, pixmap.size().width() + 32, pixmap.size().height() + 250)
            self.update_all_sockets()

