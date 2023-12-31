import os
from qtpy import QtCore, QtGui
from qtpy import QtWidgets

# from ..ainodes_backend.model_loader import ModelLoader
# from ..ainodes_backend import torch_gc

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget


from ainodes_frontend import singleton as gs
from backend_helpers.torch_helpers.model_loader import ModelLoader
from backend_helpers.torch_helpers.torch_gc import torch_gc

# from ..ainodes_backend.sd_optimizations.sd_hijack import valid_optimizations

OP_NODE_TORCH_LOADER = get_next_opcode()

class TorchLoaderWidget(QDMNodeContentWidget):
    def initUI(self):
        self.create_widgets()
        self.create_main_layout(grid=1)

    def create_widgets(self):
        checkpoint_folder = gs.prefs.checkpoints

        os.makedirs(checkpoint_folder, exist_ok=True)
        print(checkpoint_folder)
        checkpoint_files = []
        for root, dirs, files in os.walk(checkpoint_folder):
            for f in files:
                if f.endswith(('.ckpt', '.pt', '.bin', '.pth', '.safetensors')):
                    full_path = os.path.join(root, f)
                    checkpoint_files.append(full_path.replace(checkpoint_folder, ""))

        #checkpoint_files = [f for f in os.listdir(checkpoint_folder) if f.endswith(('.ckpt', '.pt', '.bin', '.pth', '.safetensors'))]
        self.dropdown = self.create_combo_box(checkpoint_files, "Model:")
        if checkpoint_files == []:
            self.dropdown.addItem("Please place a model in models/checkpoints")
            print(f"TORCH LOADER NODE: No model file found at {os.getcwd()}/models/checkpoints,")
            print(f"TORCH LOADER NODE: please download your favorite ckpt before Evaluating this node.")

        # config_folder = "models/configs"
        # config_files = [f for f in os.listdir(config_folder) if f.endswith((".yaml"))]
        # config_files = sorted(config_files, key=str.lower)
        # self.config_dropdown = self.create_combo_box(config_files, "Config:")
        # self.config_dropdown.setCurrentText("v1-inference_fp16.yaml")

        vae_folder = gs.prefs.vae

        os.makedirs(vae_folder, exist_ok=True)

        vae_files = [f for f in os.listdir(vae_folder) if f.endswith(('.ckpt', '.pt', '.bin', '.pth', '.safetensors'))]
        vae_files = sorted(vae_files, key=str.lower)
        self.vae_dropdown = self.create_combo_box(vae_files, "Vae")
        self.vae_dropdown.addItem("default")
        self.vae_dropdown.setCurrentText("default")
        #self.optimization = self.create_combo_box(["None"] + valid_optimizations, "LDM Optimization")

        self.force_reload = self.create_check_box("Force Reload")



class CenterExpandingSizePolicy(QtWidgets.QSizePolicy):
    def __init__(self, parent=None):
        super().__init__(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.parent = parent
        self.setHorizontalStretch(0)
        self.setVerticalStretch(0)
        self.setRetainSizeWhenHidden(True)
        self.setHorizontalPolicy(QtWidgets.QSizePolicy.Expanding)
        self.setVerticalPolicy(QtWidgets.QSizePolicy.Expanding)


@register_node(OP_NODE_TORCH_LOADER)
class TorchLoaderNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/torch.png"
    op_code = OP_NODE_TORCH_LOADER
    op_title = "Torch Loader"
    content_label_objname = "torch_loader_node"
    category = "aiNodes Base/Model Loading"
    input_socket_name = ["EXEC"]
    # output_socket_name = ["EXEC"]
    custom_output_socket_name = ["VAE", "CLIP", "MODEL", "EXEC"]
    def __init__(self, scene):
        super().__init__(scene, inputs=[1], outputs=[4,4,4,1])
        self.loader = ModelLoader()

    def initInnerClasses(self):
        self.content = TorchLoaderWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.icon = self.icon
        self.grNode.thumbnail = QtGui.QImage(self.grNode.icon).scaled(64, 64, QtCore.Qt.KeepAspectRatio)

        self.grNode.width = 340
        self.grNode.height = 300
        self.content.setMinimumHeight(140)
        self.content.setMinimumWidth(340)
        self.content.eval_signal.connect(self.evalImplementation)

        self.model = None
        self.clip = None
        self.vae = None

        self.loaded_sd = ""
    def remove(self):
        self.clean_sd()
        super().remove()
    def clean_sd(self):

        try:
            self.model.model.cpu()
            self.clip.cpu()
            self.vae.cpu()
        except:
            pass
        del self.model
        del self.clip
        del self.vae

        self.model = None
        self.clip = None
        self.vae = None

        torch_gc()


    def evalImplementation_thread(self, index=0):
        self.busy = True
        model_name = self.content.dropdown.currentText()
        inpaint = True if "inpaint" in model_name else False
        m = "sd_model" if not inpaint else "inpaint"

        if model_name not in gs.models:

            gs.models[model_name] = {}
            gs.models[model_name]["model"], gs.models[model_name]["clip"], gs.models[model_name]["vae"], gs.models[model_name]["clipvision"] = self.loader.load_checkpoint_guess_config(model_name, style="None")
            self.loaded_sd = model_name
            self.scene.getView().parent().window().update_models_signal.emit()
        return [gs.models[model_name]["vae"], gs.models[model_name]["clip"], gs.models[model_name]["model"]]

        # if self.loaded_sd != model_name or self.content.force_reload.isChecked() == True:
        #     self.clean_sd()
        #     self.model, self.clip, self.vae, self.clipvision = self.loader.load_checkpoint_guess_config(model_name, style="None")
        #     self.loaded_sd = model_name
        # if self.content.vae_dropdown.currentText() != 'default':
        #     model = self.content.vae_dropdown.currentText()
        #     self.vae = self.loader.load_vae(model)
        #     self.loaded_vae = model
        # else:
        #     self.loaded_vae = 'default'
        # if self.loaded_vae != self.content.vae_dropdown.currentText():
        #     model = self.content.vae_dropdown.currentText()
        #     self.vae = self.loader.load_vae(model)
        #     self.loaded_vae = model
        # return [self.vae, self.clip, self.model]



