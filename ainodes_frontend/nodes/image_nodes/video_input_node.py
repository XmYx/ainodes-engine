import time

import cv2
from PIL import Image
from qtpy import QtGui, QtCore
from qtpy.QtWidgets import QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog
from qtpy.QtCore import Qt

from ainodes_frontend.base.qimage_ops import tensor_image_to_pixmap, pil2tensor, pixmap_to_tensor

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget


OP_NODE_VIDEO_INPUT = get_next_opcode()
class VideoInputWidget(QDMNodeContentWidget):
    def initUI(self):
        self.create_widgets()
        self.create_main_layout(grid=1)
    def create_widgets(self):
        self.video = VideoPlayer()
        self.current_frame = 0
        self.label = self.create_label("")
        self.label.setAlignment(Qt.AlignCenter)
        self.skip_frames = self.create_spin_box("Skip Frames", 0, 4096, 0, 1)
        self.load_button = QPushButton("Load Video", self)
        self.load_button.clicked.connect(self.loadVideo)
        self.play_button = QPushButton("Play", self)
        self.play_button.clicked.connect(self.playVideo)
        self.stop_button = QPushButton("Stop", self)
        #self.stop_button.clicked.connect(self.stopVideo)
        self.create_button_layout([self.load_button, self.play_button])
    def advance_frame(self):
        pixmap = self.video.get_frame()
        self.label.setPixmap(pixmap)
        self.node.resize()

    def loadVideo(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.avi *.mkv)")

        if file_name:
            #print(file_name)
            self.video.load_video(file_name)
            self.advance_frame()

    def playVideo(self):
        self.advance_frame()

    def pauseVideo(self):
        pass

    def stopVideo(self):
        #self.movie.stop()
        self.current_frame = 0
        self.label.setText("Frame: 0")


@register_node(OP_NODE_VIDEO_INPUT)
class VideoInputNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/video_input.png"
    op_code = OP_NODE_VIDEO_INPUT
    op_title = "Video Input"
    content_label_objname = "video_input_node"
    category = "base/video"
    input_socket_name = ["EXEC"]
    output_socket_name = ["EXEC", "IMAGE"]


    def __init__(self, scene):
        super().__init__(scene, inputs=[1], outputs=[5,1])
        self.content.eval_signal.connect(self.evalImplementation)

    def initInnerClasses(self):
        self.content = VideoInputWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.icon = self.icon
        self.grNode.thumbnail = QtGui.QImage(self.grNode.icon).scaled(64, 64, QtCore.Qt.KeepAspectRatio)

        #self.resize()
        self.grNode.height = 200
        self.grNode.width = 200

        #self.content.setGeometry(0, 0, 512, 512)
        self.content.stop_button.clicked.connect(self.content.video.reset)
        self.markInvalid(True)
    def evalImplementation_thread(self, index=0):
        time.sleep(0.1)
        skip = self.content.skip_frames.value()
        pixmap = self.content.video.get_frame(skip=skip)
        if pixmap != None:
            self.setOutput(0, [pixmap_to_tensor(pixmap)])
            self.markDirty(False)
            self.markInvalid(False)
            self.content.label.setPixmap(pixmap)
            if len(self.getOutputs(1)) > 0:
                self.executeChild(output_index=1)
        return pixmap

    def resize(self):
        self.content.setMinimumHeight(self.content.label.pixmap().size().height())
        self.content.setMinimumWidth(self.content.label.pixmap().size().width())
        self.grNode.height = self.content.label.pixmap().size().height() + 96
        self.grNode.width = self.content.label.pixmap().size().width() + 64

        for socket in self.outputs + self.inputs:
            socket.setSocketPosition()
        self.updateConnectedEdges()



class VideoPlayer:
    def __init__(self):
        self.video_file = None
        self.video_capture = None
    def load_video(self, video_file):
        try:
            self.video_capture.release()
        except:
            pass
        self.video_file = video_file
        self.video_capture = cv2.VideoCapture(self.video_file)
    def __del__(self):
        self.video_capture.release()

    def get_frame(self, skip=1):
        # Skip frames based on the specified interval
        for _ in range(skip - 1):
            self.video_capture.grab()

        # Read the next frame and convert it to a pixmap
        ret, frame = self.video_capture.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        pixmap = pil2tensor(image)
        pixmap = tensor_image_to_pixmap(pixmap)

        # Return the pixmap if the read was successful
        if ret:
            return pixmap
        else:
            return None

    def reset(self):
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)