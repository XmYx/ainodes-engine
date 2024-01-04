import time

import numpy as np
from PIL import Image
from qtpy import QtCore, QtGui

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend.base.qimage_ops import pixmap_to_tensor, tensor_image_to_pixmap, tensor2pil, pil2tensor
from backend_helpers.torch_helpers.RIFE.infer_rife import RIFEModel


OP_NODE_RIFE = get_next_opcode()

from ainodes_frontend import singleton as gs

class RIFEWidget(QDMNodeContentWidget):
    def initUI(self):
        self.create_widgets()
        self.create_main_layout(grid=1)
    def create_widgets(self):
        self.text_label = self.create_label("Image Operators")
        self.exp = self.create_spin_box("exp", 1, 1000, 5, 1)
        self.ratio = self.create_double_spin_box("ratio", 0.00, 1.00, 0.01, 0.00)
        self.rthreshold = self.create_double_spin_box("rthreshold", 0.00, 100.00, 0.01, 0.02)
        self.rmaxcycles = self.create_spin_box("rmaxcycles", 1, 4096, 8, 1)




@register_node(OP_NODE_RIFE)
class RIFENode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/rife.png"
    op_code = OP_NODE_RIFE
    op_title = "RIFE"
    content_label_objname = "rife_node"
    category = "base/video"
    dim = (220, 320)
    NodeContent_class = RIFEWidget



    def __init__(self, scene):
        super().__init__(scene, inputs=[5,5,1], outputs=[5,1,1])
        self.painter = QtGui.QPainter()

        self.rife_temp = []
        self.content.eval_signal.connect(self.evalImplementation)
        if "rife" not in gs.models:
            gs.models["rife"] = RIFEModel()
        self.iterating = False
        pass

    # def initInnerClasses(self):
    #     self.content = RIFEWidget(self)
    #     self.grNode = CalcGraphicsNode(self)
    #     self.grNode.icon = self.icon
    #     self.grNode.thumbnail = QtGui.QImage(self.grNode.icon).scaled(64, 64, QtCore.Qt.KeepAspectRatio)
    #
    #     self.output_socket_name = ["EXEC", "EXEC/F", "IMAGE"]
    #     self.input_socket_name = ["EXEC", "IMAGE1", "IMAGE2"]
    #
    #     #self.grNode.height = 220

    #@QtCore.Slot()
    def evalImplementation_thread(self, index=0):
        exp = self.content.exp.value()
        ratio = self.content.ratio.value()
        if ratio == 0.0:
            ratio = None
        rthreshold = self.content.rthreshold.value()
        rmaxcycles = self.content.rmaxcycles.value()

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
        if pixmap1 != None and pixmap2 != None:
            image1 = tensor2pil(pixmap1)
            image2 = tensor2pil(pixmap2)
            np_image1 = np.array(image1)
            np_image2 = np.array(image2)
            frames = gs.models["rife"].infer(image1=np_image1, image2=np_image2, exp=exp, ratio=ratio, rthreshold=rthreshold, rmaxcycles=rmaxcycles)
            print(f"RIFE NODE:  {len(frames)}")
            for frame in frames:
                image = Image.fromarray(frame)
                pixmap = pil2tensor(image)
                self.setOutput(0, pixmap)
                if len(self.getOutputs(1)) > 0:
                    self.executeChild(output_index=1)
                time.sleep(0.05)
            self.markDirty(False)
            self.markInvalid(False)
        elif pixmap1 != None:
            try:
                image = tensor2pil(pixmap1[0])
                np_image = np.array(image)
                self.rife_temp.append(np_image)

                if len(self.rife_temp) == 2:
                    frames = gs.models["rife"].infer(image1=self.rife_temp[0], image2=self.rife_temp[1], exp=exp, ratio=ratio, rthreshold=rthreshold, rmaxcycles=rmaxcycles)
                    print(f"RIFE NODE:  {len(frames)}")
                    self.rife_temp = [self.rife_temp[1]]
                    return frames

                #self.setOutput(0, pixmap2)
                print(f"RIFE NODE: Using only First input")
            except:
                if len(self.getOutputs(2)) > 0:
                    self.executeChild(output_index=2)

                pass
        elif pixmap1 != None:
            try:
                self.setOutput(0, pixmap1)
                print(f"RIFE NODE: Using only Second input - Passthrough")
                if len(self.getOutputs(2)) > 0:
                    self.executeChild(output_index=2)

            except:

                pass
        if len(self.getOutputs(2)) > 0:
            self.executeChild(output_index=2)
        pass
        return None
    #@QtCore.Slot(object)
    def onWorkerFinished(self, return_frames):
        self.setOutput(0, return_frames)
        if len(self.getOutputs(1)) > 0:
            self.executeChild(output_index=1)
        pass

    def iterate_frames(self, frames):
        for frame in frames:
            image = Image.fromarray(frame)
            pixmap = tensor_image_to_pixmap(image)
            self.setOutput(0, [pixmap])
            node = self.getOutputs(1)[0]
            node.eval()
            time.sleep(0.1)
            while node.busy == True:
                time.sleep(0.1)

    def onMarkedDirty(self):
        self.value = None
    def eval(self, index=0):
        self.markDirty()
        self.content.eval_signal.emit()
