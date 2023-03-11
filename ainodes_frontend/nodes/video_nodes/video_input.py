import cv2
from PIL import Image
from qtpy.QtWidgets import QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog
from qtpy.QtCore import Qt
from ainodes_frontend.nodes.base.node_config import register_node, get_next_opcode
from ainodes_frontend.nodes.base.ai_node_base import CalcNode, CalcGraphicsNode
from ainodes_backend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_backend.node_engine.utils import dumpException
from ainodes_backend.qops import pil_image_to_pixmap

OP_NODE_VIDEO_INPUT = get_next_opcode()
class VideoInputWidget(QDMNodeContentWidget):
    def initUI(self):
        self.video = VideoPlayer()
        self.current_frame = 0

        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)

        self.load_button = QPushButton("Load Video", self)
        self.load_button.clicked.connect(self.loadVideo)

        self.play_button = QPushButton("Play", self)
        self.play_button.clicked.connect(self.playVideo)

        #self.pause_button = QPushButton("Pause", self)
        #self.pause_button.clicked.connect(self.pauseVideo)

        self.stop_button = QPushButton("Stop", self)
        #self.stop_button.clicked.connect(self.stopVideo)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        #layout.addWidget(self.movie)
        layout.addWidget(self.load_button)

        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.play_button)
        #buttons_layout.addWidget(self.pause_button)
        buttons_layout.addWidget(self.stop_button)
        layout.addLayout(buttons_layout)

        self.setLayout(layout)

    def serialize(self):
        res = super().serialize()
        return res

    def deserialize(self, data, hashmap={}):
        res = super().deserialize(data, hashmap)
        try:
            return True & res
        except Exception as e:
            dumpException(e)
        return res
    def advance_frame(self):
        pixmap = self.video.get_frame()
        self.label.setPixmap(pixmap)
        self.node.resize()

    def loadVideo(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.avi *.mkv)")

        if file_name:
            print(file_name)
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

    def onFrameChanged(self, frame):
        pixmap = self.movie.currentPixmap()
        self.node.setOutput(0, pixmap)
        self.current_frame = frame
        self.label.setPixmap(pixmap)
        #self.label.setText(f"Frame: {self.current_frame}")
        #self.node.eval()


@register_node(OP_NODE_VIDEO_INPUT)
class VideoInputNode(CalcNode):
    icon = "icons/in.png"
    op_code = OP_NODE_VIDEO_INPUT
    op_title = "Video Input"
    content_label_objname = "video_input_node"
    category = "debug"
    input_socket_name = ["EXEC"]
    output_socket_name = ["EXEC", "IMAGE"]


    def __init__(self, scene):
        super().__init__(scene, inputs=[1], outputs=[5,1])
        #self.eval()
        #self.content.eval_signal.connect(self.evalImplementation)

    def initInnerClasses(self):
        self.content = VideoInputWidget(self)
        self.grNode = CalcGraphicsNode(self)
        #self.resize()
        self.grNode.height = 200
        self.grNode.width = 200

        #self.content.setGeometry(0, 0, 512, 512)
        self.content.stop_button.clicked.connect(self.content.video.reset)
        self.markInvalid(True)
    def evalImplementation(self, index=0):
        pixmap = self.content.video.get_frame()
        if pixmap != None:
            self.setOutput(0, pixmap)
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

    def eval(self):
        self.markDirty(True)
        self.evalImplementation()


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

    def get_frame(self):
        ret, frame = self.video_capture.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(frame)
        pixmap = pil_image_to_pixmap(image)
        if ret:
            return pixmap
        else:
            return None

    def reset(self):
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)