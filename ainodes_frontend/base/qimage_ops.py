import io

import numpy as np
import torch
from PIL import Image
from qtpy.QtGui import QPixmap
from qtpy.QtCore import QBuffer
from PIL.ImageQt import ImageQt


pixmap_composite_method_list = ['blend', 'composite', 'source_over', 'destination_over', 'clear',
                               'destination', 'source_in', 'destination_in', 'source_out',
                               'destination_out', 'source_atop', 'destination_atop', 'xor',
                               'overlay', 'screen', 'soft_light', 'hard_light', 'color_dodge',
                               'color_burn', 'darken', 'lighten', 'exclusion', 'contrast']

def tensor_image_to_pixmap(tensor_image, input_type="tensor"):

    if isinstance(tensor_image, torch.Tensor):
        pil_image = tensor2pil(tensor_image)
    elif isinstance(tensor_image, np.ndarray):
        pil_image = Image.fromarray(tensor_image)
    else:
        pil_image = tensor_image

    # Convert the PIL Image object to a QImage object
    imageqt = ImageQt(pil_image)
    qimage = imageqt.convertToFormat(ImageQt.Format_RGBA8888)
    # Convert the QImage object to a QPixmap object
    pixmap = QPixmap.fromImage(qimage)
    return pixmap

def pixmap_to_tensor(pixmap, return_pil=False):
    #print(type(pixmap))
    # Convert the QPixmap object to a QImage object
    image = pixmap.toImage()
    # Convert the QImage object to a PIL Image object
    buffer = QBuffer()
    buffer.open(QBuffer.ReadWrite)
    image.save(buffer, "PNG")
    pil_image = Image.open(io.BytesIO(buffer.data()))

    if return_pil:
        return pil_image

    tensor = pil2tensor(pil_image)

    return tensor



def tensor2pil(image):
    if image is not None:
        return Image.fromarray(np.clip(255. * image.detach().numpy().squeeze(), 0, 255).astype(np.uint8))
    else:
        return None


# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)