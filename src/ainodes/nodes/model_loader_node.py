import datetime
import json
import os

from PIL.PngImagePlugin import PngInfo
from PyQt5.QtCore import QMimeData, QBuffer
from PyQt5.QtWidgets import QFileDialog
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
from modules.flux_core import model_manager

from nodeeditor.node_content_widget import QDMNodeContentWidget

# from PIL import Image






class ModelLoaderWidget(QDMNodeContentWidget):

    def initUI(self):

        self.create_main_layout()



@register_node()
class ModelLoaderNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/image_preview.png"
    op_title = "Load Model"
    content_label_objname = "ModelLoaderNode"
    category = "base/image"
    # output_data_ports = [0]
    NodeContent_class = ModelLoaderWidget
    dim = (600, 600)
    custom_output_socket_names = ['vae', 'clip', 'model', 'exec']
    make_dirty = True

    def __init__(self, scene):
        super().__init__(scene, self.op_title, inputs=[1], outputs=[9,9,9,1])
    def evalImplementation_thread(self, index=0):
        clip, vae, model = model_manager.load_model(
            (
                '/home/mix/Playground/ComfyUI/models/clip/clip_l.safetensors',
                '/home/mix/Playground/ComfyUI/models/clip/t5xxl_fp8_e4m3fn.safetensors'
            ),
            '/home/mix/Playground/ComfyUI/models/vae/flux-ae.safetensors',
            '/home/mix/Playground/ComfyUI/models/unet/flux1-dev.safetensors'
        )
        model_manager.refresh_list.emit()
        model_manager.refresh_nodes.emit()
        return [vae, clip, model]