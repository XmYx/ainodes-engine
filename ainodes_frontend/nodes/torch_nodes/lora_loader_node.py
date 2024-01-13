import os

import requests
from qtpy.QtCore import QObject, Signal
from qtpy import QtWidgets, QtCore, QtGui

from backend_helpers.torch_helpers.hash import sha256


from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend import singleton as gs

OP_NODE_LORA_LOADER = get_next_opcode()
class LoraLoaderWidget(QDMNodeContentWidget):
    def initUI(self):
        self.create_widgets()
        self.create_main_layout(grid=1)
    def create_widgets(self):
        lora_folder = gs.prefs.loras

        os.makedirs(lora_folder, exist_ok=True)

        lora_files = [f for f in os.listdir(lora_folder) if f.endswith(('.safetensors', '.ckpt', '.pt', '.bin', '.pth'))]

        self.dropdown = self.create_combo_box(lora_files, "Lora")
        if lora_files == []:
            self.dropdown.addItem("Please place a lora in models/loras")
            print(f"LORA LOADER NODE: No model file found at {os.getcwd()}/models/loras,")
            print(f"LORA LOADER NODE: please download your favorite ckpt before Evaluating this node.")
        self.force_load = self.create_check_box("Force Load")
        self.model_weight = self.create_double_spin_box("Model Weight", 0.0, 10.0, 0.1, 1.0)
        self.clip_weight = self.create_double_spin_box("Clip Weight", 0.0, 10.0, 0.1, 1.0)

        self.help_prompt = self.create_label("Trained Words:")

class CenterExpandingSizePolicy(QtWidgets.QSizePolicy):
    def __init__(self, parent=None):
        super().__init__(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.parent = parent
        self.setHorizontalStretch(0)
        self.setVerticalStretch(0)
        self.setRetainSizeWhenHidden(True)
        self.setHorizontalPolicy(QtWidgets.QSizePolicy.Expanding)
        self.setVerticalPolicy(QtWidgets.QSizePolicy.Expanding)
        #self.parent.setAlignment(Qt.AlignCenter)


class APIHandler(QObject):
    response_received = Signal(dict)

    def get_response(self, hash_value):
        url = f"https://civitai.com/api/v1/model-versions/by-hash/{hash_value}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            self.response_received.emit(data)
        else:
            # Handle error
            self.response_received.emit({})


@register_node(OP_NODE_LORA_LOADER)
class LoraLoaderNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/lora.png"
    op_code = OP_NODE_LORA_LOADER
    op_title = "Lora Loader"
    content_label_objname = "lora_loader_node"
    category = "base/lora"
    custom_input_socket_name = ["CLIP", "MODEL", "EXEC"]
    custom_output_socket_name = ["CLIP", "MODEL", "EXEC"]

    output_socket_name = ["EXEC"]
    def __init__(self, scene):
        super().__init__(scene, inputs=[4,4,1], outputs=[4,4,1])
        #self.loader = ModelLoader()

    def initInnerClasses(self):
        self.content = LoraLoaderWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.icon = self.icon
        self.grNode.thumbnail = QtGui.QImage(self.grNode.icon).scaled(64, 64, QtCore.Qt.KeepAspectRatio)

        self.grNode.width = 340
        self.grNode.height = 300
        self.content.setMinimumWidth(320)
        self.content.eval_signal.connect(self.evalImplementation)
        self.current_lora = ""
        self.apihandler = APIHandler()
        self.loaded_lora = None

    def evalImplementation_thread(self, index=0):

        clip = self.getInputData(0)
        unet = self.getInputData(1)
        assert clip is not None, "CLIP model not found, please make sure to load a torch model and connect it's outputs."
        assert unet is not None, "UNET model not found, please make sure to load a torch model and connect it's outputs."

        file = self.content.dropdown.currentText()

        sha = sha256(os.path.join(gs.prefs.loras, file))

        print("SHA", sha)

        self.apihandler.response_received.connect(self.handle_response)
        self.apihandler.get_response(sha)

        force = None if self.content.force_load.isChecked() == False else True

        strength_model = self.content.model_weight.value()
        strength_clip = self.content.clip_weight.value()

        data = {"m_w": strength_model,
                "m_C": strength_clip
                }

        #if self.values != data or self.current_lora != file:
            # if not force:
            #     unet.unpatch_model()
            #     clip.patcher.unpatch_model()
            # new_unet, new_clip = self.load_lora_to_ckpt(file, unet, clip)
        if self.values != data or self.current_lora != file:

            unet, clip = self.load_lora(unet, clip, file, strength_model, strength_clip)
        self.current_lora = file
        self.values = data


        """if gs.loaded_loras == []:
            self.current_lora = ""
        if self.current_lora != file or force:
            if file not in gs.loaded_loras or force:
                self.load_lora_to_ckpt(file)
                if file not in gs.loaded_loras:
                    gs.loaded_loras.append(file)
                self.current_lora = file"""
        return [clip, unet]
    #@QtCore.Slot(object)
    def handle_response(self, data):
        # Process the received data
        if "trainedWords" in data:

            words = "\n".join(data["trainedWords"])
            self.content.help_prompt.setText(f"Trained Words:\n{words}")

    def onInputChanged(self, socket=None):
        pass
    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        from comfy.sd import load_lora_for_models
        from comfy.utils import load_torch_file

        if strength_model == 0 and strength_clip == 0:
            return model, clip

        lora_path = os.path.join(gs.prefs.loras, lora_name)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                del self.loaded_lora

        if lora is None:
            lora = load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora, clip_lora = load_lora_for_models(model, clip, lora, strength_model, strength_clip)
        return model_lora, clip_lora

    def load_lora_to_ckpt(self, lora_name, unet, clip):
        lora_path = os.path.join(gs.prefs.loras, lora_name)
        strength_model = self.content.model_weight.value()
        strength_clip = self.content.clip_weight.value()
        unet, clip = load_lora_for_models(lora_path, strength_model, strength_clip, unet, clip)
        return unet, clip
