import copy

import torch
from PIL.ImageQt import ImageQt
from Qt import QtWidgets, QtCore, QtGui
from einops import rearrange
from PIL import Image
class TensorPreviewLayout(QtWidgets.QWidget):
    set_image_signal = QtCore.Signal(object)
    def __init__(self, parent=None):
        super(TensorPreviewLayout, self).__init__(parent)

        self.image_label = QtWidgets.QLabel("Tensor:")
        self.image = QtWidgets.QLabel()

        self.set_image_signal.connect(self.set_image)
        layout = QtWidgets.QVBoxLayout(self)

        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self.image_label)
        layout.addWidget(self.image)

        self.latent_rgb_factors = torch.tensor([
            #   R        G        B
            [0.298, 0.207, 0.208],  # L1
            [0.187, 0.286, 0.173],  # L2
            [-0.158, 0.189, 0.264],  # L3
            [-0.184, -0.271, -0.473],  # L4
        ], dtype=torch.float, device='cpu')

    def set_value(self, value):
        self.set_image_signal.emit(value)
    @QtCore.Slot(object)
    def set_image(self, input_latent):
        #latent = copy.deepcopy(input_latent)
        qImage = self.tensor_to_qimage(input_latent)
        self.image.setPixmap(QtGui.QPixmap().fromImage(QtGui.QImage(qImage)))

    def execute(self, input_latent):
        self.set_image_signal(input_latent)

    def tensor_to_qimage(self, latent):
        #mid = latent[0]
        latent = latent.float()
        latent = torch.einsum('...lhw,lr -> ...rhw', latent[0], self.latent_rgb_factors)
        latent = (((latent + 1) / 2)
                     .clamp(0, 1)  # change scale from -1..1 to 0..1
                     .mul(0xFF)  # to 0..255
                     .byte())
        # Copying to cpu as numpy array
        latent = rearrange(latent, 'c h w -> h w c').cpu().numpy()
        pil_img = Image.fromarray(latent)
        qimage = ImageQt(pil_img)
        return qimage

