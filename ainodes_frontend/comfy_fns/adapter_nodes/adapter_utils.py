import gc
import secrets

import numpy as np
from qtpy.QtCore import Signal, QObject
from qtpy.QtWidgets import QSpinBox, QDoubleSpinBox, QComboBox
from qtpy import QtCore

from ainodes_frontend import singleton as gs
if gs.torch_available:

    import torch
from PIL import Image
from qtpy.QtWidgets import QTextEdit, QLineEdit

#from ai_nodes.ainodes_engine_base_nodes.ainodes_backend import pixmap_to_tensor, tensor_image_to_pixmap
if gs.torch_available:
    loadBackup = torch.load

#from . import install_all_comfy_nodes


from ainodes_frontend.base import register_node, AiNode, register_node_now, get_next_opcode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget


default_numeric = {"FLOAT":{"min":0.0,
                     "max":100.0,
                     "default":1.0,
                     "step":0.01},
            "INT":  {"min":-1,
                     "max":100,
                     "default":1,
                     "step":1},
            "NUMBER":{"min":-1,
                     "max":1,
                     "default":1,
                     "step":1},
            }


data_subtypes = ["STRING", "PROMPT", "NUMBER", "FLOAT"]

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)




def create_node(node_class, node_name, ui_inputs, inputs, input_names, outputs, output_names, category_input, fn=None):



    class_name = node_name.replace(" ", "")
    class_code = get_next_opcode()

    # Create new Widget class
    class Widget(QDMNodeContentWidget):
        finished = Signal()

        def get_widget(self, widget_type):
            widget_types = {"INT":self.create_spin_box,
                            "FLOAT":self.create_double_spin_box,
                            "STRING":self.create_line_edit,
                            "MULTISTRING":self.create_text_edit
                            }
            if widget_type in widget_types:
                return widget_types[widget_type], True
            else:
                return self.create_line_edit, False

        def create_widget(self, widget_name, widget_type, widget_params):

            widget, known = self.get_widget(widget_type)

            if known:
                if widget_type in default_numeric.keys():
                    numeric_defaults = default_numeric[widget_type]
                    min_val = widget_params.get('min', numeric_defaults['min'])
                    max_val = widget_params.get('max', numeric_defaults['max'])
                    def_val = widget_params.get('default', numeric_defaults['default'])
                    step_val = widget_params.get('step', numeric_defaults['step'])
                    if widget_type == "INT":
                        min_val = int(min_val)
                        max_val = int(max_val)
                        def_val = int(def_val)
                        step_val = int(step_val)
                    min_val = max(min_val, -2147483647)
                    max_val = min(max_val, 2147483647)

                    setattr(self, widget_name,
                            widget(label_text=widget_name, min_val=min_val, max_val=max_val, step=step_val,
                                   default_val=def_val))
                else:
                    default = widget_params.get('default', 'default_placeholder')
                    multiline = widget_params.get('multiline', False)

                    if widget_type == 'PROMPT':
                        multiline = True
                        default = "Prompt"

                    if multiline:
                        setattr(self, widget_name, self.create_text_edit(widget_name, default=str(default)))
                    else:
                        setattr(self, widget_name,
                                self.create_line_edit(widget_name, default=str(default), placeholder=str(default)))
            else:
                setattr(self, widget_name, widget)

            # Connect the created_widget's signal to mark_node_dirty
            for created_widget in getattr(self, 'widget_list', []):
                if isinstance(created_widget, (QSpinBox, QDoubleSpinBox)):
                    created_widget.valueChanged.connect(self.mark_node_dirty)
                elif isinstance(created_widget, (QLineEdit, QTextEdit)):
                    created_widget.textChanged.connect(self.mark_node_dirty)
                elif isinstance(created_widget, QComboBox):
                    created_widget.currentIndexChanged.connect(self.mark_node_dirty)

        @QtCore.Slot()
        def mark_node_dirty(self, value=None):
            # print("marking")
            self.node.markDirty(True)

        def initUI(self):
            for item in ui_inputs:
                #print(item)
                tp = item[1][0]
                if type(tp) == str:
                    # source = item[1]
                    name = item[0]
                    widget_type = item[1][0]
                    widget_params = item[1][1]
                    self.create_widget(name, widget_type, widget_params)
                elif type(tp) == list:
                    combobox_name = item[0]
                    combobox_items = item[1][0]
                    setattr(self, combobox_name,
                            self.create_combo_box(combobox_items, combobox_name, accessible_name=combobox_name))
                else:
                    pass
                    #print("OTHER ITEM", type(tp), item)

            self.create_main_layout(grid=1)


    # Create new Node class
    class Node(AiNode, node_class):

        icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
        help_text = "Data objects in aiNodes are simple dictionaries,\n" \
                    "that can hold any values under any name.\n" \
                    "In most cases, you'll find them drive parameters,\n" \
                    "or hold sequences of images. For an example, the\n" \
                    "OpenAI node emits it's prompt in a data line,\n" \
                    "but you'll find this info in all relevant places."
        op_code = class_code
        op_title = node_name
        content_label_objname = class_name.lower().replace(" ", "_")
        # print("Comfy content_label_objname", content_label_objname)
        category = f"{category_input}/{node_class.CATEGORY if hasattr(node_class, 'CATEGORY') else 'Diffusers'}"#"WAS NODES"
        NodeContent_class = Widget
        dim = (340, 180)
        output_data_ports = outputs
        exec_port = len(outputs)

        custom_input_socket_name = input_names
        custom_output_socket_name = output_names

        make_dirty = False if 'loader' in node_name.lower() else True

        def __init__(self, scene):
            super().__init__(scene, inputs=inputs, outputs=outputs)
            self.fn = fn if fn else getattr(node_class, node_class.FUNCTION, None)

            self.output_names = output_names
            self.output_types = outputs

            self.exec_port = len(self.outputs) - 1

            modifier = len(inputs)
            if len(outputs) > len(inputs):
                modifier = len(outputs)

            self.content.setGeometry(0, 15, self.content.geometry().width(), self.content.geometry().height())

            self.grNode.height = self.content.geometry().height() + (modifier * 30)

            self.update_all_sockets()

            self.adapted_inputs = (input_names, ui_inputs)
            self.adapted_outputs = output_names

            self.cache = {}

            self.device = gs.device

        @torch.inference_mode()
        def evalImplementation_thread(self):

            data_inputs = {}
            x = 0

            for input in self.adapted_inputs[0]:
                data = None
                if input != "EXEC":
                    data = self.getInputData(x)

                    # print(data, input.lower())

                if data is not None:
                    data_inputs[input.lower()] = data
                x += 1

            for ui_input in self.adapted_inputs[1]:
                input_type = ui_input[1][0]
                input_name = ui_input[0]
                widget = getattr(self.content, input_name)

                if isinstance(input_type, str):
                    if input_type in default_numeric.keys():
                        data = widget.value()
                        data_inputs[ui_input[0].lower()] = data
                    else:
                        multiline = False
                        if "multiline" in ui_input[1][1]:
                            multiline = ui_input[1][1]
                        if isinstance(widget, QTextEdit):
                            data = widget.toPlainText()
                        elif isinstance(widget, QLineEdit):
                            data = widget.text()
                        if data != "":
                            data_inputs[ui_input[0].lower()] = data
                elif isinstance(input_type, list):
                    data = widget.currentText()
                    if data != "":
                        data_inputs[ui_input[0].lower()] = data
            all_ins = self.getAllInputs()
            #print("COMFY ALL INPUTS", all_ins)
            if "exec" in all_ins:
                del all_ins["exec"]
            data_inputs.update(all_ins)
            # # Create a unique key for the current set of inputs
            # cache_key = str(sorted(data_inputs.items()))
            #
            # # Check if the result is already in the cache
            # if cache_key in self.cache:
            #     result = self.cache[cache_key]
            # else:
            #     # If not in cache, compute the result and store it in the cache
            #     result = self.fn(self, **data_inputs)
            #     self.cache[cache_key] = result
            # del data_inputs
            if "seed" in data_inputs:
                if data_inputs.get("seed", 0) == 0:
                    data_inputs["seed"] = secrets.randbelow(2147483647)
                    self.markDirty(True)
            if self.make_dirty:
                self.markDirty(True)
            if self.isDirty():
                with torch.no_grad():
                    result = self.fn(self, **data_inputs)

                x = 0

                for i in list(self.adapted_outputs):
                    if i != "EXEC":
                        self.setOutput(x, result[x])
                    x += 1
                self.markDirty(False)
            self.content.update()
            self.content.finished.emit()
            #gc.collect()

            return True

        def onWorkerFinished(self, result, exec=True):
            self.busy = False
            self.markDirty(False)
            if exec:
                if result:
                    self.executeChild(self.exec_port)

        def can_run(self):
            if not self.inputs:
                return True

            def is_dirty_connected_node(edge, attribute):
                socket = getattr(edge, attribute, None)
                if socket and hasattr(socket, 'node') and socket.node != self:
                    return socket.node.isDirty()
                return False

            for socket in self.inputs:
                for edge in socket.edges:
                    if is_dirty_connected_node(edge, 'start_socket') or is_dirty_connected_node(edge, 'end_socket'):
                        #print(f"Node {self} cannot run because connected node {socket.node} is dirty.")
                        return False

            return True

        def onInputChanged(self, socket=None):
            self.markDirty(True)

        def remove(self):
            # Delete attributes that were set during self.setOutput
            for index in range(len(self.outputs)):
                object_name = self.getID(index)
                if hasattr(self, object_name):
                    # If the value is a torch tensor and is on the GPU, delete it
                    value = getattr(self, object_name)
                    if isinstance(value, torch.Tensor) and value.is_cuda:
                        value.cpu()
                        del value
                        torch.cuda.empty_cache()
                    else:
                        try:
                            value.cpu()
                        except:
                            try:
                                value.to("cpu")
                            except:
                                pass
                        del value
                        gc.collect()
                    # Delete the attribute itself
                    delattr(self, object_name)

            # Clear the cache dictionary
            self.cache.clear()
            super().remove()

    register_node_now(class_code, Node)



def get_node_parameters(node_class):

    ordered_inputs = []

    for key, value in node_class.INPUT_TYPES().items():
        for value_name, value_params in value.items():



            ordered_inputs.append((value_name, value_params))

    return ordered_inputs


def parse_comfynode(node_name, node_class):
    node_content_class = node_name.lower().replace(" ", "_")

    # print(node_class.INPUT_TYPES())
    try:
        ordered_inputs = get_node_parameters(node_class)

        # print("ORDERED INPUTS #1", ordered_inputs)

        inputs = []
        input_names = []
        outputs = []
        for i in ordered_inputs:
            if i[1][0] == "LATENT":
                inputs.append(2)
            elif i[1][0] in ["IMAGE", "MASK"]:
                inputs.append(5)
            elif i[1][0] == "CONDITIONING":
                inputs.append(3)
            elif i[1][0] in ["EXTRA_PNGINFO", "EXTRA_SETTINGS", "DISCO_DIFFUSION_EXTRA_SETTINGS"]:
                inputs.append(6)
            elif i[1][0] in ["VAE", "CLIP", "MODEL", "GUIDED_DIFFUSION_MODEL", "GUIDED_CLIP"]:
                inputs.append(4)
            if i[1][0] in ["LATENT", "IMAGE", "MASK", "CONDITIONING", "EXTRA_PNGINFO", "VAE", "CLIP", "MODEL", "GUIDED_DIFFUSION_MODEL", "GUIDED_CLIP", "EXTRA_SETTINGS", "DISCO_DIFFUSION_EXTRA_SETTINGS"]:
                input_names.append(i[1][0])
        input_names.append("EXEC")
        ordered_inputs.append(input_names)
        # elif i[1][0] == "IMAGE_BOUNDS":
        #     inputs.append(6)
        # print("RESULT INPUTS", inputs)

        fn = getattr(node_class, node_class.FUNCTION)
        ordered_outputs = []
        output_names = []
        x = 0
        for i in node_class.RETURN_TYPES:
            data = {}
            data['type'] = i
            data['name'] = "DEFAULT"
            if hasattr(node_class, "RETURN_NAMES"):
                data['name'] = node_class.RETURN_NAMES[x]
            if i in ["STRING", "NUMBER", "EXTRA_SETTINGS", "DISCO_DIFFUSION_EXTRA_SETTINGS"]:
                outputs.append(6)
            elif i == "LATENT":
                outputs.append(2)
            elif i in ["IMAGE", "MASK"]:
                outputs.append(5)
            elif i == "CONDITIONING":
                outputs.append(3)
            elif i in ["VAE", "CLIP", "MODEL", "GUIDED_DIFFUSION_MODEL", "GUIDED_CLIP"]:
                outputs.append(4)
            if i in ["LATENT", "IMAGE", "MASK", "MASKS", "CONDITIONING", "EXTRA_PNGINFO", "VAE", "CLIP", "MODEL", "GUIDED_DIFFUSION_MODEL", "GUIDED_CLIP",
                     "STRING", "NUMBER", "EXTRA_SETTINGS", "DISCO_DIFFUSION_EXTRA_SETTINGS"]:
                output_names.append(i)
            ordered_outputs.append(data)
        output_names.append('EXEC')

        # print("CREATED OUTPUT NAMES", output_names)

        ordered_outputs.append(output_names)
        outputs.append(1)
        inputs.append(1)
        # Use the function
        create_node(node_class, node_name, ordered_inputs, inputs, ordered_outputs, outputs, fn=fn)
    except:
        pass
