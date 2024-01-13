import numpy as np
import torch
from qtpy import QtWidgets, QtCore, QtGui

from backend_helpers.cnet_preprocessors.refonly.hook import ControlModelType, ControlParams, UnetHook
from ...base.qimage_ops import pixmap_to_tensor, tensor2pil

from ainodes_frontend import singleton as gs
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ..image_nodes.image_op_node import HWC3

OP_NODE_CN_APPLY = get_next_opcode()

model_free_preprocessors = [
    "reference_only",
    "reference_adain",
    "reference_adain+attn"
]


class CNApplyWidget(QDMNodeContentWidget):
    def initUI(self):
        self.create_widgets()
        self.create_main_layout(grid=1)
        self.main_layout.setContentsMargins(15, 15, 15, 25)
    def create_widgets(self):
        # self.strength = self.create_double_spin_box("Strength", 0.00, 100.00, 0.01, 1.00)
        # self.cfg_scale = self.create_double_spin_box("Guidance Scale", 0.01, 100.00, 0.01, 7.5)
        self.start = self.create_spin_box("Start", 0, 100, 0)
        self.stop = self.create_spin_box("End", 0, 100, 100)
        self.tresh_a = self.create_spin_box("Treshold a", 64, 1024, 512, 64)
        self.tresh_b = self.create_spin_box("Treshold b", 64, 1024, 512, 64)
        # self.soft_injection = self.create_check_box("Soft Inject")
        # self.cfg_injection = self.create_check_box("CFG Inject")
        self.cleanup_on_run = self.create_check_box("CleanUp on Run", True)
        self.control_net_selector = self.create_combo_box(["controlnet", "t2i", "reference"], "Control Style")
        self.model_free_selector = self.create_combo_box(model_free_preprocessors, "Modelfree Style")
        self.button = QtWidgets.QPushButton("Run")
        self.cleanup_button = QtWidgets.QPushButton("CleanUp")
        self.create_button_layout([self.button, self.cleanup_button])

@register_node(OP_NODE_CN_APPLY)
class CNApplyNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/apply_cnet.png"
    op_code = OP_NODE_CN_APPLY
    op_title = "Apply ControlNet"
    content_label_objname = "CN_apply_node"
    category = "base/controlnet"
    custom_input_socket_name = ["IMAGE", "LATENT", "MODEL", "COND", "EXEC"]
    NodeContent_class = CNApplyWidget
    dim = (600, 400)

    def __init__(self, scene):
        super().__init__(scene, inputs=[5,2,4,3,1], outputs=[4,2,3,1])

        self.content.button.clicked.connect(self.evalImplementation)
        self.content.cleanup_button.clicked.connect(self.clean)
        #pass
        self.latest_network = None
        self.patched = False
        # Create a worker object
    # def initInnerClasses(self):
    #     self.content = CNApplyWidget(self)
    #     self.grNode = CalcGraphicsNode(self)
    #     self.grNode.icon = self.icon
    #     self.grNode.thumbnail = QtGui.QImage(self.grNode.icon).scaled(64, 64, QtCore.Qt.KeepAspectRatio)
    #
    #     self.grNode.height = 600
    #     self.grNode.width = 256
    #     self.content.setMinimumWidth(256)
    #     self.content.setMinimumHeight(420)
    #     self.content.eval_signal.connect(self.evalImplementation)

    def clean(self):
        if self.latest_network is not None:
            try:
                self.latest_network.restore(gs.models["sd"].model.model.diffusion_model)
            except:
                pass

    #@QtCore.Slot()
    def evalImplementation_thread(self, index=0):
        self.markDirty(True)
        #self.markInvalid(True)
        image_list = self.getInputData(0)
        latent = self.getInputData(1)
        model = self.getInputData(2)
        cond = self.getInputData(3)

        return_list = []
        style = self.content.control_net_selector.currentText()
        # weight = self.content.strength.value()
        # cfg_scale = self.content.cfg_scale.value()

        if style == "reference":

            model, latent = self.ref_only(model, latent, 1)
            return [model, latent, cond]

        else:
            cond_node, index = self.getInput(1)
            conditioning_list = cond_node.getOutput(index)
            if len(conditioning_list) == 1:
                for image in image_list:
                    result = self.add_control_image(conditioning_list[0], image)
                    return_list.append(result)
            elif len(conditioning_list) == len(image_list):
                x = 0
                for image in image_list:
                    result = self.add_control_image(conditioning_list[x], image)
                    return_list.append(result)
        return [model, latent, return_list]

    # def onMarkedDirty(self):
    #     self.value = None
    def ref_only(self, model, reference, batch_size):

        #model_reference = model.clone()
        size_latent = list(reference["samples"].shape)
        size_latent[0] = batch_size
        latent = {}
        latent["samples"] = torch.zeros(size_latent)

        batch = latent["samples"].shape[0] + reference["samples"].shape[0]

        def reference_apply(q, k, v, extra_options):
            k = k.clone().repeat(1, 2, 1)
            offset = 0
            if q.shape[0] > batch:
                offset = batch

            for o in range(0, q.shape[0], batch):
                for x in range(1, batch):
                    k[x + o, q.shape[1]:] = q[o, :]

            return q, k, k

        self.set_model_patch(model, reference_apply, "attn1_patch", self.id)
        out_latent = torch.cat((reference["samples"], latent["samples"]))

        if "noise_mask" in latent:
            mask = latent["noise_mask"]
        else:
            mask = torch.ones((64, 64), dtype=torch.float32, device="cpu")

        if len(mask.shape) < 3:
            mask = mask.unsqueeze(0)
        if mask.shape[0] < latent["samples"].shape[0]:
            print(latent["samples"].shape, mask.shape)
            mask = mask.repeat(latent["samples"].shape[0], 1, 1)

        out_mask = torch.zeros((1, mask.shape[1], mask.shape[2]), dtype=torch.float32, device="cpu")
        return model, {"samples": out_latent, "noise_mask": torch.cat((out_mask, mask))}
    def set_model_patch(self, model, patch, name, uid):
        to = model.model_options["transformer_options"]
        if "patches" not in to:
            to["patches"] = {}

        if not self.patched:
            to["patches"][name] = to["patches"].get(name, []) + [patch]
            self.patch_index = len(to["patches"][name]) - 1
        else:
            to["patches"][name][self.patch_index] = patch



    def apply_ref_control(self, image, weight, cfg_scale, start=0, stop=100, soft_injection=True, cfg_injection=True, model=None):

        # gs.models["sd"].model.model.start_control = start
        # gs.models["sd"].model.model.stop_control = stop
        vae = self.getInputData(3)
        cleanup = self.content.cleanup_on_run.isChecked()
        if cleanup == True:
            if self.latest_network is not None:
                try:
                    self.latest_network.restore(model.model.diffusion_model)
                except:
                    pass

        #unet = gs.models["sd"].model.model.diffusion_model

        # gs.models["sd"].model.cuda()

        model_net = None

        image = tensor2pil(image)

        processor_res = int(image.size[0] // 8)

        image = np.array(image) / 255.0

        image = image.astype(np.uint8)

        image = HWC3(image)

        image = torch.from_numpy(image)[None,]
        # c = []
        control_hint = image.movedim(-1, 1).to("cuda")

        # input_image = HWC3(np.asarray(input_image))

        # control = detected_map

        control_model_type = ControlModelType.AttentionInjection

        forward_params = []


        model_free_preprocessors = [
            "reference_only",
            "reference_adain",
            "reference_adain+attn"
        ]

        model_net = dict(
            name=self.content.model_free_selector.currentText(),
            preprocessor_resolution=processor_res,
            threshold_a=self.content.tresh_a.value(),
            threshold_b=self.content.tresh_b.value()
        )


        forward_param = ControlParams(
            control_model=model_net,
            hint_cond=control_hint,
            weight=weight,
            guidance_stopped=False,
            start_guidance_percent=start,
            stop_guidance_percent=stop,
            advanced_weighting=None,
            control_model_type=control_model_type,
            global_average_pooling=False,
            hr_hint_cond=None,
            batch_size=1,
            instance_counter=0,
            is_vanilla_samplers=False,
            cfg_scale=cfg_scale,
            soft_injection=soft_injection,
            cfg_injection=cfg_injection,
        )
        forward_params.append(forward_param)

        del model_net
        self.latest_network = UnetHook(lowvram=False)
        self.latest_network.hook(model=model.model.diffusion_model, sd_ldm=model, control_params=forward_params, vae=vae)
        return "Done"

    def add_control_image(self, conditioning, image, progress_callback=None):
        start = self.content.start.value()
        stop = self.content.stop.value()

        # gs.models["sd"].model.model.start_control = start
        # gs.models["sd"].model.model.stop_control = stop

        # image = pixmap_to_tensor(image)

        # image = np.array(image).astype(np.float32) / 255.0
        #
        #
        # image = torch.from_numpy(image)[None,]
        c = []
        control_hint = image.movedim(-1,1)
        for t in conditioning:
            n = [t[0], t[1].copy()]
            n[1]['control_hint'] = control_hint
            n[1]['control_strength'] = self.content.strength.value()
            c.append(n)
        return c

