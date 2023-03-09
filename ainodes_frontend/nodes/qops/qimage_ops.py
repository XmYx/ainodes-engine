import io

from PIL import Image
from qtpy.QtGui import QPixmap
from qtpy.QtCore import QBuffer
from PIL.ImageQt import ImageQt

def pil_image_to_pixmap(pil_image):
    # Convert the PIL Image object to a QImage object
    imageqt = ImageQt(pil_image)
    qimage = imageqt.convertToFormat(ImageQt.Format_RGBA8888)
    # Convert the QImage object to a QPixmap object
    pixmap = QPixmap.fromImage(qimage)
    return pixmap


def pixmap_to_pil_image(pixmap):
    #print(type(pixmap))
    # Convert the QPixmap object to a QImage object
    image = pixmap.toImage()
    # Convert the QImage object to a PIL Image object
    buffer = QBuffer()
    buffer.open(QBuffer.ReadWrite)
    image.save(buffer, "PNG")
    pil_image = Image.open(io.BytesIO(buffer.data()))
    return pil_image