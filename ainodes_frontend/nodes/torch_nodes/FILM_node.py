import copy

import numpy as np
import torch
from PIL import Image
from qtpy import QtWidgets, QtCore, QtGui

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.base.qimage_ops import pixmap_to_tensor, tensor_image_to_pixmap, tensor2pil, pil2tensor
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from deforum import FilmModel


OP_NODE_FILM = get_next_opcode()

from ainodes_frontend import singleton as gs

class FILMWidget(QDMNodeContentWidget):
    def initUI(self):
        # Create a label to display the image
        self.film = self.create_spin_box("FRAMES", 1, 4096, 10, 1)
        self.create_main_layout(grid=1)



@register_node(OP_NODE_FILM)
class FILMNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/film.png"
    op_code = OP_NODE_FILM
    op_title = "FILM"
    content_label_objname = "FILM_node"
    category = "base/video"
    make_dirty = True

    def __init__(self, scene):
        super().__init__(scene, inputs=[5,5,1], outputs=[5,1])
        self.painter = QtGui.QPainter()

        self.FILM_temp = []
        self.content.eval_signal.connect(self.evalImplementation)
        if "FILM" not in gs.models:
            gs.models["FILM"] = FilmModel()
        pass
        #self.eval()
    def __del__(self):
        if "FILM" in gs.models:
            print("Cleaned FILM")
            del gs.models["FILM"]

    def initInnerClasses(self):
        self.content = FILMWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.icon = self.icon
        self.grNode.thumbnail = QtGui.QImage(self.grNode.icon).scaled(64, 64, QtCore.Qt.KeepAspectRatio)

        self.output_socket_name = ["EXEC", "EXEC/F", "IMAGE"]
        self.input_socket_name = ["EXEC", "IMAGE1", "IMAGE2"]

        self.grNode.height = 220
        self.content.eval_signal.connect(self.evalImplementation)


    #@QtCore.Slot()
    def evalImplementation_thread(self):
        return_frames = []
        if "FILM" not in gs.models:
            gs.models["FILM"] = FilmModel()

        if self.getInput(1) != None:
            node, index = self.getInput(1)
            pixmap1 = node.getOutput(index)
        else:
            pixmap1 = None
        if self.getInput(0) != None:
            node, index = self.getInput(0)
            pixmap2 = node.getOutput(index)
        else:
            pixmap2 = None
        #print("FILM NODE", pixmap1, pixmap2)
        if pixmap1 != None and pixmap2 != None:
            image1 = tensor2pil(pixmap1[0])
            image2 = tensor2pil(pixmap2[0])
            np_image1 = np.array(image1)
            np_image2 = np.array(image2)
            frames = gs.models["FILM"].inference(np_image1, np_image2, inter_frames=25)
            #print(f"FILM NODE:  {len(frames)}")
            skip_first, skip_last = True, False
            if skip_first:
                frames.pop(0)
            if skip_last:
                frames.pop(-1)

            for frame in frames:
                image = Image.fromarray(frame)
                pixmap = tensor_image_to_pixmap(image)
                return_frames.append(pixmap)
        elif pixmap1 != None:
            #for pixmap in pixmap1:
            image = tensor2pil(pixmap1)
            np_image = np.array(image.convert("RGB"))
            self.FILM_temp.append(np_image)
            if len(self.FILM_temp) == 2:
                frames = gs.models["FILM"].inference(self.FILM_temp[0], self.FILM_temp[1], inter_frames=self.content.film.value())
                skip_first, skip_last = True, False
                if skip_first:
                    frames.pop(0)
                if skip_last:
                    frames.pop(-1)

                for frame in frames:
                    image = Image.fromarray(copy.deepcopy(frame))
                    pixmap = pil2tensor(image)
                    return_frames.append(pixmap)
                self.FILM_temp = [self.FILM_temp[1]]
            print(f"[ FILM NODE: Created {len(return_frames)} frames ]")
        if len(return_frames) > 0:
            return_frames = torch.stack(return_frames, dim=0)

            return [return_frames]
        else:
            return [pixmap1]
    def clearOutputs(self):
        self.FILM_temp = []
        super().clearOutputs()
    # def iterate_frames(self, frames):
    #     self.iterating = True
    #     for frame in frames:
    #         node = None
    #         if len(self.getOutputs(1)) > 0:
    #             node = self.getOutputs(1)[0]
    #         if node is not None:
    #             image = Image.fromarray(copy.deepcopy(frame))
    #             pixmap = tensor_image_to_pixmap(image)
    #             self.setOutput(0, pixmap)
    #             node.eval()
    #     self.iterating = False
    # def onMarkedDirty(self):
    #     self.value = None


