import contextlib
import os

import requests
import torch
from qtpy.QtCore import Qt
from qtpy.QtCore import QObject, Signal
#from qtpy.QtGui import Qt
from qtpy import QtWidgets, QtCore, QtGui

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.base.settings import handle_ainodes_exception
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget

from ainodes_frontend import singleton as gs

from qtpy.QtWidgets import QDialog, QListWidget, QCheckBox, QDoubleSpinBox, QVBoxLayout, QDialogButtonBox, \
    QListWidgetItem, QHBoxLayout, QWidget

from backend_helpers.torch_helpers.vram_management import offload_to_device


# from ai_nodes.ainodes_engine_base_nodes.ainodes_backend import get_torch_device
# from ai_nodes.ainodes_engine_base_nodes.ainodes_backend.hash import sha256


class EmbedDialog(QDialog):
    def __init__(self, embed_files, prev_dict):
        super().__init__()
        self.embed_files = embed_files
        self.embed_values = {}

        self.setWindowTitle("Select Embeddings")
        layout = QVBoxLayout()
        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget)

        for file_name in embed_files:
            item_widget = QWidget()
            item_layout = QHBoxLayout(item_widget)

            check_box = QCheckBox()
            check_box.setText(file_name)
            item_layout.addWidget(check_box)

            spin_box = QDoubleSpinBox()
            spin_box.setRange(0.0, 1.0)
            spin_box.setSingleStep(0.1)
            spin_box.setEnabled(True)
            item_layout.addWidget(spin_box)
            for item in prev_dict:
                #print(item)
                if item['embed']['filename'] == file_name:
                    check_box.setChecked(True)
                    spin_box.setValue(item['embed']['value'])
            self.d = prev_dict
            check_box.stateChanged.connect(lambda state, box=spin_box: box.setEnabled(state == Qt.Checked))
            self.embed_values[file_name] = [check_box, spin_box]

            list_widget_item = QListWidgetItem()
            list_widget_item.setSizeHint(item_widget.sizeHint())  # Set size hint for proper layout
            self.list_widget.addItem(list_widget_item)
            self.list_widget.setItemWidget(list_widget_item, item_widget)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        self.setLayout(layout)

    def accept(self):
        selected_embeds = []
        for file_name, check_box in self.embed_values.items():
            if check_box[0].isChecked():
                selected_embeds.append({"embed":{"filename":file_name,
                                        "value":check_box[1].value(),
                                        "word":""}})
        self.selected_embeds = selected_embeds
        super().accept()
        return selected_embeds
OP_NODE_CONDITIONING_XL = get_next_opcode()
class ConditioningXLWidget(QDMNodeContentWidget):
    def initUI(self):
        self.create_widgets()
        self.create_main_layout(grid=1)
    def create_widgets(self):
        # self.create_text_edit("Prompt", spawn="prompt")





        # # self.create_text_edit("prompt")
        self.create_text_edit("Prompt", placeholder="Prompt or Negative Prompt (use 2x Conditioning Nodes for Stable Diffusion),\n"
                                                                  "and connect them to a K Sampler.\n"
                                                                  "If you want to control your resolution,\n"
                                                                  "or use an init image, use an Empty Latent Node.", spawn='prompt_l')

        self.create_text_edit("Prompt 2", placeholder="Prompt or Negative Prompt (use 2x Conditioning Nodes for Stable Diffusion),\n"
                                                                  "and connect them to a K Sampler.\n"
                                                                  "If you want to control your resolution,\n"
                                                                  "or use an init image, use an Empty Latent Node.", spawn='prompt_g')

        self.width_val = self.create_spin_box("Height", min_val=256, max_val=4096, default_val=1024)
        self.height_val = self.create_spin_box("Height", min_val=256, max_val=4096, default_val=1024)
        self.crop_w = self.create_spin_box("Crop Width", min_val=0, max_val=4096, default_val=0)
        self.crop_h = self.create_spin_box("Crop Height", min_val=0, max_val=4096, default_val=0)
        self.target_width = self.create_spin_box("Target Width", min_val=256, max_val=4096, default_val=1024)
        self.target_height = self.create_spin_box("Target Height", min_val=256, max_val=4096, default_val=1024)

        self.skip = self.create_spin_box("Clip Skip", min_val=-11, max_val=0, default_val=-1)
        self.embed_checkbox = self.create_check_box("Use embeds")
        self.button = QtWidgets.QPushButton("Get Conditioning")
        self.set_embeds = QtWidgets.QPushButton("Embeddings")
        self.create_button_layout([self.button, self.set_embeds])

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

@register_node(OP_NODE_CONDITIONING_XL)
class ConditioningXLAiNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/conditioning.png"
    op_code = OP_NODE_CONDITIONING_XL
    op_title = "Conditioning XL"
    content_label_objname = "cond_ainode_xl"
    category = "base/sampling"

    custom_input_socket_name = ["CLIP", "DATA", "EXEC"]
    NodeContent_class = ConditioningXLWidget
    dim = (400, 800)

    def __init__(self, scene):
        super().__init__(scene, inputs=[4,6,1], outputs=[3,6,1])
        self.content.button.clicked.connect(self.evalImplementation)
        self.content.set_embeds.clicked.connect(self.show_embeds)
        self.embed_dict = []
        self.apihandler = APIHandler()
        self.apihandler.response_received.connect(self.handle_response)
        self.string = ""
        self.clip_skip = -1
        self.device = gs.device
        if self.device in [torch.device('mps'), torch.device('cpu')]:
            self.context = contextlib.nullcontext()
        else:
            self.context = torch.autocast(gs.device.type)




    def show_embeds(self):

        embed_files = [f for f in os.listdir(gs.prefs.embeddings) if f.endswith(('.ckpt', '.pt', '.bin', '.pth', '.safetensors'))]
        if embed_files is not []:
            # The embedding strings returned as: "embedding:<filename without extension>:<weight> where weight is a float between 0.0 and 1.0"
            self.show_embed_dialog(embed_files)


    def show_embed_dialog(self, embed_files):
        dialog = EmbedDialog(embed_files, self.embed_dict)
        if dialog.exec() == QDialog.Accepted:
            selected_embeds = dialog.selected_embeds
            self.embed_dict = selected_embeds

            """for embed in self.embed_dict:
                print("word", embed["embed"]["filename"])
                file = os.path.join(gs.prefs.embeddings, embed["embed"]["filename"])
                sha = sha256(file)
                self.apihandler.response_received.connect(self.handle_response)
                self.apihandler.get_response(sha)"""

        else:
            return None


    #@QtCore.Slot()
    def evalImplementation_thread(self, index=0, prompt_override=None):
        clip = self.getInputData(0)
        assert clip is not None, "Please make sure to load a model, and connect it's clip output to the input"


        width = self.content.width_val.value()
        height = self.content.height_val.value()
        crop_h = self.content.crop_h.value()
        crop_w = self.content.crop_w.value()


        try:
            result = None
            data = None
            prompt_l = self.content.prompt_l.toPlainText()
            prompt_g = self.content.prompt_g.toPlainText()
            if prompt_override is not None:
                prompt = prompt_override

            string = ""
            for item in self.embed_dict:
                string = f'{string} embedding:{item["embed"]["filename"]}'

            string = "" if not self.content.embed_checkbox.isChecked() else string

            prompt_l = f"{prompt_l} {string}" if (self.content.embed_checkbox.isChecked and string != "") else prompt_l
            data = self.getInputData(1)
            if data:
                if "prompt_l" in data:
                    prompt = data["prompt"]
                else:
                    data["prompt_l"] = prompt_l
                if "model" in data:
                    if data["model"] == "deepfloyd_1":
                        result = [gs.models["deepfloyd_1"].encode_prompt(prompt_l)]
                else:
                    if prompt_override is not None:
                        prompt = prompt_override
                    result = self.get_conditioning(text_l=prompt_l, clip=clip)

            else:
                data = {}
                data["prompt_l"] = prompt_l
                result = self.get_conditioning(text_l=prompt_l,
                                               text_g=prompt_g,
                                               width=width,
                                               height=height,
                                               clip=clip,
                                               crop_h=crop_h,
                                               crop_w=crop_w)
            if gs.logging:
                print(f"CONDITIONING NODE: Applying conditioning with prompt: {prompt_l}")
            return [result, data]
        except Exception as e:
            done = handle_ainodes_exception()
            if type(e) is KeyError and 'clip' in str(e):
                print("Clip / SD Model not loaded yet, please place and validate a Torch loader node")
            else:
                print(repr(e))
            return [None]

    #@QtCore.Slot(object)
    def handle_response(self, data):
        if 'files' in data:
            file = data['files'][0]['name']

            for item in self.embed_dict:
                if item['embed']['filename'] == file:
                    item['embed']['word'] = "\n".join(data["trainedWords"])
                    item['embed']['word'] = item['embed']['filename']
        string = ""
        for item in self.embed_dict:
            if item['embed']['word'] != "":
                string = f'{string} embedding:{item["embed"]["word"]}'
        self.string = string

    def get_conditioning(self,
                 text_g:str="",
                 text_l:str="",
                 width=512,
                 height=512,
                 crop_w=0,
                 crop_h=0,
                 target_width=512,
                 target_height=512,
                 clip=None,
                 progress_callback=None):
        offload_to_device(clip, gs.device)

        tokens = clip.tokenize(text_g)
        tokens["l"] = clip.tokenize(text_l)["l"]
        if len(tokens["l"]) != len(tokens["g"]):
            empty = clip.tokenize("")
            while len(tokens["l"]) < len(tokens["g"]):
                tokens["l"] += empty["l"]
            while len(tokens["l"]) > len(tokens["g"]):
                tokens["g"] += empty["g"]
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)

        if gs.vram_state in ["low", "medium"]:
            offload_to_device(clip, "cpu")
        return [[cond,{"pooled_output": pooled, "width": width, "height": height, "crop_w": crop_w, "crop_h": crop_h,
                   "target_width": target_width, "target_height": target_height}]]

    #@QtCore.Slot(object)
    # def onWorkerFinished(self, result, exec=True):
    #     self.busy = False
    #     #super().onWorkerFinished(None)
    #     if result is not None:
    #         self.setOutput(1, result[0])
    #         self.setOutput(0, result[1])
    #         self.markDirty(False)
    #         self.markInvalid(False)
    #         if gs.should_run:
    #             self.executeChild(2)
