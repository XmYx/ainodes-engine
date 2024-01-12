import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from qtpy import QtWidgets, QtCore, QtGui

from backend_helpers.cnet_preprocessors import hed
from backend_helpers.cnet_preprocessors.mlsd import MLSDdetector
from backend_helpers.cnet_preprocessors.midas import MidasDetector
from backend_helpers.cnet_preprocessors import OpenposeDetector

from backend_helpers.torch_helpers.semseg.semseg_inference import SemSegModel
from deforum.utils.deforum_framewarp_utils import anim_frame_warp_3d
from ...base.qimage_ops import pixmap_to_tensor, tensor_image_to_pixmap, pil2tensor, tensor2pil

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget

from ainodes_frontend import singleton as gs
OP_NODE_IMAGE_OPS = get_next_opcode()

image_ops_methods = [
    "resize",
    "canny",
    "tile_preprocess",
    "fake_scribble",
    'hed',
    'depth',
    'normal',
    'mlsd',
    'openpose',
    "autocontrast",
    "Brightness",
    "Contrast",
    "Sharpness",
    "colorize",
    "contrast",
    "grayscale",
    "invert",
    "mirror",
    "posterize",
    "solarize",
    "flip",
    "depth_transform",
    "semseg",
    "antialias"
]
image_ops_valid_methods = [
    "Brightness",
    "Contrast",
    "Sharpness"
]

class ImageOpsWidget(QDMNodeContentWidget):
    def initUI(self):
        self.create_widgets()
        self.create_main_layout(grid=1)
        self.dropdown.currentIndexChanged.connect(self.dropdownChanged)
        self.height_value.valueChanged.connect(self.calculate_image_ratio)
        self.width_value.valueChanged.connect(self.calculate_image_ratio)
        self.dropdownChanged()

    def create_widgets(self):
        self.text_label = QtWidgets.QLabel("Image Operator:")
        self.dropdown = self.create_combo_box(image_ops_methods, "Image Operator:")

        self.enhancer_level = self.create_double_spin_box("Enhance level", min_val=0.0, max_val=10.0, default_val=1.0, step=0.1)

        self.width_value = self.create_spin_box("Width:", 64, 4096, 512, 64)
        self.height_value = self.create_spin_box("Height:", 64, 4096, 512, 64)

        self.resize_ratio_label = self.create_label("Resize Ratio")

        self.canny_low = self.create_spin_box("Canny Low:", 0, 255, 100, 1)
        self.canny_high = self.create_spin_box("Canny High:", 0, 255, 100, 1)

        self.midas_a = self.create_double_spin_box("Midas A:", 0.00, 100.00, 0.01, np.pi * 2.0)
        self.midas_bg = self.create_double_spin_box("Midas Bg:", 0.00, 100.00, 1.00, 0.01)


    def dropdownChanged(self, event=None):
        value = self.dropdown.currentText()
        if value == 'resize':
            self.width_value.setVisible(True)
            self.height_value.setVisible(True)
            self.resize_ratio_label.setVisible(True)
            self.canny_high.setVisible(False)
            self.canny_low.setVisible(False)
            self.midas_a.setVisible(False)
            self.midas_bg.setVisible(False)
        elif value == 'canny':
            self.width_value.setVisible(False)
            self.height_value.setVisible(False)
            self.resize_ratio_label.setVisible(False)
            self.canny_high.setVisible(True)
            self.canny_low.setVisible(True)
            self.midas_a.setVisible(False)
            self.midas_bg.setVisible(False)
        elif value in ['depth', 'normal', 'mlsd']:
            self.width_value.setVisible(False)
            self.height_value.setVisible(False)
            self.resize_ratio_label.setVisible(False)
            self.canny_high.setVisible(False)
            self.canny_low.setVisible(False)
            self.midas_a.setVisible(True)
            self.midas_bg.setVisible(True)

        else:
            self.width_value.setVisible(False)
            self.height_value.setVisible(False)
            self.resize_ratio_label.setVisible(False)
            self.canny_high.setVisible(False)
            self.canny_low.setVisible(False)
            self.midas_a.setVisible(False)
            self.midas_bg.setVisible(False)
    def calculate_image_ratio(self):
        width = self.width_value.value()
        height = self.height_value.value()
        ratio = width / height
        text = return_ratio_string(ratio, width, height)
        self.resize_ratio_label.setText(text)


@register_node(OP_NODE_IMAGE_OPS)
class ImageOpNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/image_op.png"
    op_code = OP_NODE_IMAGE_OPS
    op_title = "Image Operators"
    content_label_objname = "image_op_node"
    category = "base/image"

    make_dirty = True

    custom_output_socken_name = ["IMAGE", "MASK", "EXEC"]


    def __init__(self, scene):
        super().__init__(scene, inputs=[5,6,1], outputs=[5,5,1])

    def initInnerClasses(self):
        self.content = ImageOpsWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.icon = self.icon
        self.grNode.thumbnail = QtGui.QImage(self.grNode.icon).scaled(64, 64, QtCore.Qt.KeepAspectRatio)

        #self.content.dropdown.currentIndexChanged.connect(self.evalImplementation)
        self.output_socket_name = ["EXEC", "MASK","IMAGE"]
        self.input_socket_name = ["EXEC", "DATA", "IMAGE"]
        self.grNode.height = 340
        self.grNode.width = 280
        self.content.setMinimumHeight(130)
        self.content.setMinimumWidth(260)
        self.content.eval_signal.connect(self.evalImplementation)

    #@QtCore.Slot()
    def evalImplementation_thread(self):
        tensors = None
        mask = None
        return_pixmap = None
        return_tensor_list = []
        tensors = self.getInputData(0)
        method = self.content.dropdown.currentText()
        if tensors is not None:
            for tensor in tensors:
                return_tensor, mask = self.image_op(tensor, method)
                return_tensor_list.append(return_tensor)
            tensors = torch.stack(return_tensor_list)

        return [tensors, mask]


    def image_op(self, pixmap, method):
        tensor = None
        mask = None
        # Convert the QPixmap object to a PIL Image object
        image = tensor2pil(pixmap)
        if method in image_ops_valid_methods:
            # Get the requested ImageEnhance method
            enhance_method = getattr(ImageEnhance, method, None)

            if enhance_method:
                # Create an instance of the Enhancer for the PIL Image object
                enhancer = enhance_method(image)
                # Apply enhancement to the image
                image = enhancer.enhance(self.content.enhancer_level.value())
            else:
                # If the requested method is not available, raise an error
                raise ValueError(f"Invalid ImageEnhance method: {method}")
        elif method == 'antialias':
            image = antialias_image(image, int(self.content.enhancer_level.value()))

        elif method == 'resize':
            width = self.content.width_value.value()
            height = self.content.height_value.value()
            image = image.resize((width, height), resample=Image.LANCZOS)
        elif method == 'canny':

            image = np.array(image)
            image = cv2.Canny(image, self.content.canny_low.value(), self.content.canny_high.value(), L2gradient=True)
            image = HWC3(image)
            image = Image.fromarray(image)
        elif method == 'tile_preprocess':

            # def tile_resample(img, res=512, thr_a=1.0, **kwargs):
            thr_a = 1.0
            img = np.array(image)
            img = HWC3(img)
            if thr_a > 1.1:
                H, W, C = img.shape
                H = int(float(H) / float(thr_a))
                W = int(float(W) / float(thr_a))
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            image = Image.fromarray(img)


        elif method == 'fake_scribble':
            image = np.array(image)
            detector = hed.HEDdetector()
            image = detector(image)
            image = HWC3(image)
            image = hed.nms(image, 127, 3.0)
            image = cv2.GaussianBlur(image, (0, 0), 3.0)
            image[image > 4] = 255
            image[image < 255] = 0
            image = Image.fromarray(image)
            detector.netNetwork.cpu()
            detector.netNetwork = None

            del detector
        elif method == 'hed':
            # Ref: https://github.com/lllyasviel/ControlNet/blob/main/gradio_hed2image.py
            image = np.array(image)
            detector = hed.HEDdetector()
            image = detector(image)
            image = HWC3(image)
            image = Image.fromarray(image)
            detector.netNetwork.cpu()
            detector.netNetwork = None

            del detector
        elif method == 'depth':
            image = np.array(image)
            detector = MidasDetector()
            a = self.content.midas_a.value()
            bg_threshold = self.content.midas_bg.value()
            depth_map_np, normal_map_np = detector(image, a, bg_threshold)
            image = HWC3(depth_map_np)
            image = Image.fromarray(image)
            detector.model.cpu()
            detector.model = None
            del detector
        elif method == 'depth_transform':
            image = np.array(image)
            model = MidasDetector()
            model.model.cpu()
            model.model = None
            model.load_midas()

            a = self.content.midas_a.value()
            bg_threshold = self.content.midas_bg.value()
            device = "cuda"
            args = {
                    "translation_x" : 0,
                    "translation_y" : 0,
                    "translation_z" : 0,
                    "rotation_3d_x" : 25,
                    "rotation_3d_y" : 0,
                    "rotation_3d_z" : 0,
                    }
            tensor = model.predict(image)
            if self.getInput(1) != None:
                node, index = self.getInput(1)
                data = node.getOutput(index)
                for key, value in data.items():
                    if key[0] == 'Warp3D':
                        #print(key, value)
                        args[key[1]] = value
            with torch.no_grad():
                np_image, mask = anim_frame_warp_3d(device, image, tensor, args["translation_x"], args["translation_y"], args["translation_z"], args["rotation_3d_x"], args["rotation_3d_y"], args["rotation_3d_z"])
                mask = mask.cpu()
                mask = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
                # print(mask.shape)
                #print(mask.device)
            model.deforum_midas.cpu()
            del model.deforum_midas
            del model
            image = Image.fromarray(np_image)

        elif method == 'normal':
            image = np.array(image)
            detector = MidasDetector()
            a = self.content.midas_a.value()
            bg_threshold = self.content.midas_bg.value()
            depth_map_np, normal_map_np = detector(image, a, bg_threshold)
            image = HWC3(normal_map_np)
            image = Image.fromarray(image)
            detector.model.cpu()
            detector.model = None

            del detector
        elif method == 'mlsd':
            image = image.convert('RGB')
            image = np.array(image)
            #print(image.shape)
            detector = MLSDdetector()
            a = self.content.midas_a.value()
            bg_threshold = self.content.midas_bg.value()
            mlsd = detector(image, bg_threshold, a)
            image = HWC3(mlsd)
            image = Image.fromarray(image)
            detector.model.cpu()
            detector.model = None
            del detector
        elif method == 'openpose':
            image = image.convert('RGB')
            image = np.array(image)
            #print(image.shape)
            detector = OpenposeDetector()
            pose, _ = detector(image, True)
            image = HWC3(pose)
            image = Image.fromarray(image)
            del detector
        elif method == 'semseg':
            image = image.convert('RGB')
            image = np.array(image)
            #print(image.shape)
            detector = SemSegModel()
            #print(detector)
            image, masks = detector.predict(image)
            data = {
                ("images", "list") : masks
            }
            self.setOutput(1, data)
            #pose, _ = detector(image, True)
            #image = HWC3(pose)
            #image = Image.fromarray(np_image)

        elif method == 'invert':
            image = image.convert("RGB")
            image = ImageOps.invert(image)
        if image != None:
            # Convert the PIL Image object to a QPixmap object
            tensor = pil2tensor(image)[0]


        #print(tensor.shape)
        return tensor, mask




def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def return_ratio_string(ratio, width, height):
    if ratio == 1:
        return f"{width}x{height} (Square)"
    elif ratio == 1.25:
        return f"{width}x{height} (5:4, Classic Medium Format)"
    elif ratio == 1.33:
        return f"{width}x{height} (4:3, Classic Standard Format)"
    elif ratio == 1.4:
        return f"{width}x{height} (7:5, Modern Medium Format)"
    elif ratio == 1.5:
        return f"{width}x{height} (3:2, Classic 35mm Format)"
    elif ratio == 1.6:
        return f"{width}x{height} (16:10, Widescreen)"
    elif ratio == 1.66:
        return f"{width}x{height} (5:3, European Widescreen)"
    elif ratio == 1.77:
        return f"{width}x{height} (16:9, HDTV)"
    elif ratio == 1.85:
        return f"{width}x{height} (1.85:1, Widescreen Film)"
    elif ratio == 2:
        return f"{width}x{height} (2:1, Panavision)"
    elif ratio == 2.2:
        return f"{width}x{height} (11:5, IMAX)"
    elif ratio == 2.35:
        return f"{width}x{height} (2.35:1, Cinemascope)"
    elif ratio == 2.39:
        return f"{width}x{height} (2.39:1, Anamorphic)"
    elif ratio == 2.4:
        return f"{width}x{height} (12:5, Ultrawide)"
    elif ratio == 2.55:
        return f"{width}x{height} (17:7, Univisium)"
    elif ratio == 2.76:
        return f"{width}x{height} (2.76:1, Ultra Panavision)"
    elif ratio == 3:
        return f"{width}x{height} (3:1, Triptych)"
    elif ratio == 3.25:
        return f"{width}x{height} (13:4, Triplewide)"
    elif ratio == 3.5:
        return f"{width}x{height} (7:2, Triplewide)"
    elif ratio == 4/3.:
        return f"{width}x{height} (1.33:1, Classic Standard Format)"
    elif ratio == 3/2.:
        return f"{width}x{height} (1.5:1, Classic 35mm Format)"
    elif ratio == 4/5.:
        return f"{width}x{height} (4:5, Portrait Format)"
    elif ratio == 5/4.:
        return f"{width}x{height} (5:4, Classic Medium Format)"
    elif ratio == 6/5.:
        return f"{width}x{height} (6:5, Antique Format)"
    elif ratio == 5/6.:
        return f"{width}x{height} (5:6, Portrait Format)"
    elif ratio == 9/16.:
        return f"{width}x{height} (9:16, HDTV Portrait)"
    elif ratio == 10/16.:
        return f"{width}x{height} (10:16, Widescreen Portrait)"
    elif ratio == 1.6/1.33:
        return f"{width}x{height} (4:3 Anamorphic)"
    elif ratio == 1.85/1.33:
        return f"{width}x{height} (4:3 Widescreen Film)"
    elif ratio == 1.85/1.5:
        return f"{width}x{height} (3:2 Widescreen Film)"
    elif ratio == 2.2/1.33:
        return f"{width}x{height} (4:3 IMAX)"
    elif ratio == 2.35/1.33:
        return f"{width}x{height} (4:3 Cinemascope)"
    elif ratio == 2.39/1.33:
        return f"{width}x{height} (4:3 Anamorphic)"
    elif ratio == 2.76/1.33:
        return f"{width}x{height} (4:3 Ultra Panavision)"
    elif ratio == 1/1.414:
        return f"{width}x{height} (1:√2, ISO A Paper)"
    elif ratio == 1.414/1:
        return f"{width}x{height} (√2:1, ISO B Paper)"
    elif ratio == 1/1.732:
        return f"{width}x{height} (1:√3, ISO C Paper)"
    elif ratio == 1.732/1:
        return f"{width}x{height} (√3:1, ISO D Paper)"
    elif ratio == 1/2.414:
        return f"{width}x{height} (1:√6, ISO E Paper)"
    elif ratio == 2.414/1:
        return f"{width}x{height} (√6:1, ISO F Paper)"
    elif ratio == 1/1.618:
        return f"{width}x{height} (1:φ, Golden Ratio)"
    elif ratio == 1.618/1:
        return f"{width}x{height} (φ:1, Reverse Golden Ratio)"
    elif ratio == 1.33/1:
        return f"{width}x{height} (4:3, Standard 8mm and 16mm Film)"
    elif ratio == 1.37/1:
        return f"{width}x{height} (Academy Ratio, 35mm Film)"
    elif ratio == 1.66/1:
        return f"{width}x{height} (5:3, Super 16mm Film)"
    elif ratio == 1.85/1:
        return f"{width}x{height} (Widescreen, 35mm Film)"
    elif ratio == 2.35/1:
        return f"{width}x{height} (2.35:1, CinemaScope, 35mm Film)"
    elif ratio == 2.39/1:
        return f"{width}x{height} (2.39:1, Anamorphic, 35mm Film)"
    elif ratio == 3/1.33:
        return f"{width}x{height} (4:3, Standard 8mm Film)"
    elif ratio == 3/1.37:
        return f"{width}x{height} (Academy Ratio, 8mm Film)"
    elif ratio == 3/1.66:
        return f"{width}x{height} (5:3, Super 8mm Film)"
    elif ratio == 3/1.85:
        return f"{width}x{height} (Widescreen, 8mm Film)"
    elif ratio == 3/2.35:
        return f"{width}x{height} (2.35:1, CinemaScope, 8mm Film)"
    elif ratio == 3/2.39:
        return f"{width}x{height} (2.39:1, Anamorphic, 8mm Film)"
    elif ratio == 3.15/2:
        return f"{width}x{height} (1.575:1, Polavision)"
    elif ratio == 4/2.75:
        return f"{width}x{height} (1.45:1, Instamatic)"
    elif ratio == 1.8/1:
        return f"{width}x{height} (16:9, HDV)"
    elif ratio == 1.458/1:
        return f"{width}x{height} (Dutch Angle)"
    elif ratio == 1/1.17:
        return f"{width}x{height} (1:1.17, VistaVision)"
    elif ratio == 1/1.33:
        return f"{width}x{height} (4:3, Standard Television)"
    elif ratio == 1.78/1:
        return f"{width}x{height} (16:9, HDTV)"
    elif ratio == 1.6/1:
        return f"{width}x{height} (16:10, Widescreen Computer)"
    elif ratio == 1.25/1:
        return f"{width}x{height} (5:4, Computer Monitor)"
    elif ratio == 1.33/1:
        return f"{width}x{height} (4:3, Computer Monitor)"
    elif ratio == 1.43/1:
        return f"{width}x{height} (7:5, Computer Monitor)"
    elif ratio == 1.6/0.9:
        return f"{width}x{height} (16:9, Widescreen Monitor)"
    elif ratio == 1.78/1.33:
        return f"{width}x{height} (4:3, Standard Def Letterbox)"
    elif ratio == 1.78/1.5:
        return f"{width}x{height} (3:2, HD Letterbox)"
    elif ratio == 1.85/1.33:
        return f"{width}x{height} (4:3, Film Letterbox)"
    elif ratio == 2.35/1.33:
        return f"{width}x{height} (4:3, Film Pan-Scan)"
    elif ratio == 2.76/1.33:
        return f"{width}x{height} (4:3, Ultra Panavision Letterbox)"
    elif ratio == 1/1.29:
        return f"{width}x{height} (1:1.29, Two-Perf Techniscope)"
    else:
        return f"{width}x{height} ({ratio:.2f}:1)"

def antialias_image(image, antialias):

    image = np.array(image)

    orig_size = (int(image.shape[1]), int(image.shape[0]))
    # Calculate the target size
    target_size = (int(image.shape[1] * antialias), int(image.shape[0] * antialias))

    # Resize the image to the target size using cubic interpolation
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)

    # Resize the image back to the original size using Gaussian blur
    antialiased_image = cv2.GaussianBlur(resized_image, (0, 0), antialias)

    antialiased_image = cv2.resize(antialiased_image, orig_size, interpolation=cv2.INTER_NEAREST_EXACT  )


    image = Image.fromarray(antialiased_image)

    return image