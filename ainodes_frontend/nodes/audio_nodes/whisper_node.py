import threading

#from qtpy.QtWidgets import QLineEdit, QLabel, QPushButton, QFileDialog, QVBoxLayout
from qtpy import QtWidgets, QtCore

from ainodes_backend.sound_recorder import AudioRecorder
from ainodes_frontend.nodes.base.node_config import register_node, get_next_opcode
from ainodes_frontend.nodes.base.ai_node_base import CalcNode, CalcGraphicsNode
from ainodes_backend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_backend.node_engine.utils import dumpException
from ainodes_backend import singleton as gs
import whisper

SCHEDULERS = ["karras", "normal", "simple", "ddim_uniform"]
SAMPLERS = ["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral",
            "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde",
            "dpmpp_2m", "ddim", "uni_pc", "uni_pc_bh2"]

OP_CODE = get_next_opcode()
class Whisper():

    def __init__(self):

        if "whisper" not in gs.models:
            gs.models["whisper"] = whisper.load_model("base")
    def infer(self):
        # load audio and pad/trim it to fit 30 seconds
        audio = whisper.load_audio("recording.wav")
        audio = whisper.pad_or_trim(audio)

        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(gs.models["whisper"].device)

        # detect the spoken language
        _, probs = gs.models["whisper"].detect_language(mel)
        print(f"Detected language: {max(probs, key=probs.get)}")

        # decode the audio
        options = whisper.DecodingOptions()
        result = whisper.decode(gs.models["whisper"], mel, options)

        # print the recognized text
        print(result.text)
        return result.text



class WhisperWidget(QDMNodeContentWidget):
    finish_eval_signal = QtCore.Signal()
    def initUI(self):
        self.recorder = AudioRecorder()
        #audio_data = self.recorder.record()
        #self.recorder.save_to_file(audio_data)
        self.whisper = Whisper()
        self.run_button = QtWidgets.QPushButton("Run")
        self.label = QtWidgets.QTextEdit()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(15,15,15,25)
        layout.addWidget(self.label)
        layout.addWidget(self.run_button)
        #layout.addWidget(self.stop_button)
        #layout.addWidget(self.checkbox)
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


@register_node(OP_CODE)
class WhisperNode(CalcNode):
    icon = "icons/in.png"
    op_code = OP_CODE
    op_title = "Whisper"
    content_label_objname = "whisper_node"
    category = "debug"

    def __init__(self, scene):
        super().__init__(scene, inputs=[1], outputs=[1])
        self.content.run_button.clicked.connect(self.start)
        self.content.finish_eval_signal.connect(self.finishEval)
        #self.content.stop_button.clicked.connect(self.stop)

        self.interrupt = False
        # Create a worker object
    def initInnerClasses(self):
        self.content = WhisperWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.height = 340
        self.grNode.width = 256
        self.content.setMinimumWidth(256)
        self.content.setMinimumHeight(300)
        self.input_socket_name = ["EXEC"]
        self.output_socket_name = ["EXEC"]

        #self.content.setMinimumHeight(400)
        #self.content.setMinimumWidth(256)
        #self.content.image.changeEvent.connect(self.onInputChanged)

    def evalImplementation(self, index=0):
        self.markDirty(True)
        self.markInvalid(True)
        self.content.recorder.record()
        self.text = self.content.whisper.infer()
        self.content.finish_eval_signal.emit()
    @QtCore.Slot()
    def finishEval(self):
        txt = f"{self.content.label.toPlainText()}, {self.text}"
        self.content.label.setText(txt)
        if len(self.getOutputs(0)) > 0:
            self.executeChild(0)
        return None

    def onMarkedDirty(self):
        self.value = None

    def stop(self):
        self.interrupt = True
        return
    def start(self):
        self.interrupt = False
        thread0 = threading.Thread(target=self.evalImplementation, args=(0,))
        thread0.start()









