import datetime
import os

import cv2
import imageio
import numpy as np
from qtpy import QtWidgets
from qtpy.QtWidgets import QPushButton, QVBoxLayout
from ainodes_frontend.nodes.base.node_config import register_node, get_next_opcode
from ainodes_frontend.nodes.base.ai_node_base import CalcNode, CalcGraphicsNode
from ainodes_backend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_backend.node_engine.utils import dumpException
from ainodes_backend.qops import pixmap_to_pil_image

OP_NODE_VIDEO_SAVE = get_next_opcode()
class VideoOutputWidget(QDMNodeContentWidget):
    def initUI(self):
        self.video = GifRecorder()
        self.current_frame = 0

        self.save_button = QPushButton("Save buffer to GIF", self)
        #self.new_button = QPushButton("New Video", self)
        #self.save_button.clicked.connect(self.loadVideo)

        self.width_value = QtWidgets.QSpinBox()
        self.width_value.setMinimum(64)
        self.width_value.setSingleStep(64)
        self.width_value.setMaximum(4096)
        self.width_value.setValue(512)

        self.height_value = QtWidgets.QSpinBox()
        self.height_value.setMinimum(64)
        self.height_value.setSingleStep(64)
        self.height_value.setMaximum(4096)
        self.height_value.setValue(512)

        self.fps = QtWidgets.QSpinBox()
        self.fps.setMinimum(1)
        self.fps.setSingleStep(1)
        self.fps.setMaximum(4096)
        self.fps.setValue(24)


        layout = QVBoxLayout()
        #layout.addWidget(self.new_button)
        layout.addWidget(self.save_button)
        #layout.addWidget(self.width_value)
        #layout.addWidget(self.height_value)
        layout.addWidget(self.fps)

        self.setLayout(layout)

    def serialize(self):
        res = super().serialize()
        res['fps'] = self.fps.value()
        return res

    def deserialize(self, data, hashmap={}):
        res = super().deserialize(data, hashmap)
        try:
            self.fps.setValue(int(data['fps']))
            #self.image.setPixmap(value)
            return True & res
        except Exception as e:
            dumpException(e)
        return res


@register_node(OP_NODE_VIDEO_SAVE)
class VideoOutputNode(CalcNode):
    icon = "icons/in.png"
    op_code = OP_NODE_VIDEO_SAVE
    op_title = "Video Save"
    content_label_objname = "video_output_node"
    category = "debug"
    input_socket_name = ["EXEC", "IMAGE"]
    output_socket_name = ["EXEC", "IMAGE"]


    def __init__(self, scene):
        super().__init__(scene, inputs=[5,1], outputs=[5,1])
        self.filename = ""
        #self.eval()
        #self.content.eval_signal.connect(self.evalImplementation)

    def initInnerClasses(self):
        self.content = VideoOutputWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.content.save_button.clicked.connect(self.start_new_video)
        self.grNode.height = 300
        self.grNode.width = 260

        self.content.setGeometry(0, 0, 260, 230)
        self.markInvalid(True)
    def evalImplementation(self, index=0):
        if self.getInput(0) is not None:
            input_node, other_index = self.getInput(0)
            if not input_node:
                self.grNode.setToolTip("Input is not connected")
                self.markInvalid()
                return

            val = input_node.getOutput(other_index)
            image = pixmap_to_pil_image(val)
            frame = np.array(image)
            self.markInvalid(False)
            self.markDirty(True)
            self.content.video.add_frame(frame)
            self.setOutput(0, val)
            if len(self.getOutputs(1)) > 0:
                self.executeChild(output_index=1)

        return None
    def close(self):
        self.content.video.close(self.filename)
        self.markDirty(False)
        self.markInvalid(False)
        self.start_new_video()
    def resize(self):
        self.content.setMinimumHeight(self.content.label.pixmap().size().height())
        self.content.setMinimumWidth(self.content.label.pixmap().size().width())
        self.grNode.height = self.content.label.pixmap().size().height() + 96
        self.grNode.width = self.content.label.pixmap().size().width() + 64

        for socket in self.outputs + self.inputs:
            socket.setSocketPosition()
        self.updateConnectedEdges()

    def eval(self):
        self.markDirty(True)
        self.evalImplementation()

    def start_new_video(self):
        self.markDirty(True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs("output/gifs", exist_ok=True)

        self.filename = f"output/gifs/{timestamp}.gif"
        fps = self.content.fps.value()
        self.content.video.close(self.filename, fps)
        print(f"VIDEO SAVE NODE: Done. The frame buffer is now empty.")

class VideoRecorder:

    def __init__(self):
        pass
    def start_recording(self, filename, fps, width, height):
        #fourcc = cv2.VideoWriter_fourcc(*'GIF')
        self.video_writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'GIF'), fps, (width, height))
        #self.video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    def add_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.video_writer.write(frame)
        print("VIDEO SAVE NODE: New frame added to video stream")

    def close(self, filename=""):
        self.video_writer.release()
        #os.rename("test.mp4", filename)
        print(f"VIDEO SAVE NODE: Video saved as {filename}")


class GifRecorder:

    def __init__(self):
        self.frames = []

    def start_recording(self, filename, fps, width, height):
        self.filename = filename
        self.fps = fps

    def add_frame(self, frame):
        self.frames.append(frame)
        print(f"VIDEO SAVE NODE: Image added to frame buffer, current frames: {len(self.frames)}")

    def close(self, filename, fps):
        self.filename = filename
        self.fps = fps
        print(f"VIDEO SAVE NODE: Video saving {len(self.frames)} frames at {self.fps}fps as {self.filename}")
        imageio.mimsave(self.filename, self.frames, fps=self.fps)
        self.frames = []
