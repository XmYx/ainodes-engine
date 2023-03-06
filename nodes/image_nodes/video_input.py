from qtpy.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog
from qtpy.QtGui import QPixmap, QMovie
from qtpy.QtCore import Qt, QTimer, QUrl
from nodes.base.node_config import register_node, OP_NODE_VIDEO_INPUT
from nodes.base.ai_node_base import CalcNode, CalcGraphicsNode
from node_engine.node_content_widget import QDMNodeContentWidget
from node_engine.utils import dumpException
from qtpy import QtWidgets


class VideoInputWidget(QDMNodeContentWidget):
    def initUI(self):
        self.current_frame = 0
        self.movie = QMovie()
        self.movie.frameChanged.connect(self.onFrameChanged)

        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)

        self.load_button = QPushButton("Load Video", self)
        self.load_button.clicked.connect(self.loadVideo)

        self.play_button = QPushButton("Play", self)
        self.play_button.clicked.connect(self.playVideo)

        self.pause_button = QPushButton("Pause", self)
        self.pause_button.setEnabled(False)
        self.pause_button.clicked.connect(self.pauseVideo)

        self.stop_button = QPushButton("Stop", self)
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stopVideo)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        #layout.addWidget(self.movie)
        layout.addWidget(self.load_button)

        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.play_button)
        buttons_layout.addWidget(self.pause_button)
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

    def loadVideo(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.avi *.mkv)")

        if file_name:
            print(file_name)
            self.movie.setFileName(file_name)
            self.movie.jumpToFrame(5)
            pixmap = self.movie.currentPixmap()
            self.label.setPixmap(pixmap)
            self.play_button.setEnabled(True)
            self.stop_button.setEnabled(True)

    def playVideo(self):
        self.movie.start()
        self.play_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)

    def pauseVideo(self):
        self.movie.setPaused(True)
        self.play_button.setEnabled(True)
        self.pause_button.setEnabled(False)

    def stopVideo(self):
        self.movie.stop()
        self.play_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
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


    def __init__(self, scene):
        super().__init__(scene, inputs=[], outputs=[1])
        self.eval()
        self.content.eval_signal.connect(self.evalImplementation)

    def initInnerClasses(self):
        self.content = VideoInputWidget(self)
        self.grNode = CalcGraphicsNode(self)

        self.grNode.height = 512
        self.grNode.width = 512
        #self.content.movie.setMinimumHeight(512)
        #self.content.movie.setMinimumWidth(512)
        # self.content.setMinimumHeight(self.content.image.pixmap().size().height())
        # self.content.setMinimumWidth(self.content.image.pixmap().size().width())

        self.content.setGeometry(0, 0, 512, 512)

    def evalImplementation(self, index=0):
        if self.content.movie.frameCount() == 0:
            return

        pixmap = self.content.movie.currentPixmap()
        self.setOutput(0, pixmap)

        self.content.current_frame += 1
        if self.content.current_frame >= self.content.movie.frameCount():
            self.content.current_frame = 0

        self.content.movie.jumpToFrame(self.content.current_frame)
        self.content.movie.setPaused(True)
        #self.timer.start(100)
        self.markDirty(False)
        self.markInvalid(False)
        return pixmap