import os
import sys
import importlib
import traceback
import logging
import inspect
import gc
import secrets

import numpy as np
import torch
from PIL import Image
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QLineEdit
from qtpy import QtCore

from node_core.node_register import register_node, AiNode, register_node_now
from nodeeditor.node_content_widget import QDMNodeContentWidget

# Define default numeric parameters
default_numeric = {
    "FLOAT": {"min": 0.0, "max": 100.0, "default": 1.0, "step": 0.01},
    "INT": {"min": -2147483647, "max": 2147483647, "default": 1, "step": 1},
    "NUMBER": {"min": -1, "max": 1, "default": 1, "step": 1},
}

# Data subtypes for UI elements
UI_TYPES = {'INT', 'FLOAT', 'STRING', 'BOOLEAN', 'PROMPT', 'TEXT', 'NUMBER'}

# Data port types (initialized as a mutable set)
DATA_PORT_TYPES = {
    'LATENT', 'IMAGE', 'MASK', 'CONDITIONING', 'VAE', 'CLIP', 'MODEL',
    'CONTROL_NET', 'STYLE_MODEL', 'CLIP_VISION', 'CLIP_VISION_OUTPUT',
    'GLIGEN', 'UNET', 'SAMPLES', 'TIME_EMBEDDING', 'TUPLE', 'EXTRA_PNGINFO',
    'EXTRA_SETTINGS', 'DISCO_DIFFUSION_EXTRA_SETTINGS', 'HYPERNETWORK',
    'INPAINT_IMAGE', 'INPAINT_MASK', 'CLIP2', 'CLIP_MIXED',
    'VIDEO', 'AUDIO', 'MASKS', 'LATENTDIFFMASK', 'LATENTMASK',
    'LATENTDIFF', 'SDXL_CONDITIONING', 'CONDITIONS', "SIGMAS",
    "NOISE", "GUIDER", "SAMPLER", "LATENT_IMAGE", "UPSCALE_MODEL", "PHOTOMAKER",
    # Add more data types as necessary
}

# Function to convert tensor to PIL image
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# Function to convert PIL image to tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# Function to load custom nodes
def load_custom_nodes(paths):
    node_classes = {}
    for path in paths:
        if not os.path.exists(path):
            continue
        sys.path.append(path)
        for filename in os.listdir(path):
            if filename.endswith('.py') and not filename.startswith('_'):
                module_name = filename[:-3]
                module_path = os.path.join(path, filename)
                try:
                    logging.info(f"Loading custom node: {module_name}")
                    module_spec = importlib.util.spec_from_file_location(module_name, module_path)
                    module = importlib.util.module_from_spec(module_spec)
                    sys.modules[module_name] = module
                    module_spec.loader.exec_module(module)
                    if hasattr(module, 'NODE_CLASS_MAPPINGS'):
                        node_class_mappings = module.NODE_CLASS_MAPPINGS
                        node_classes.update(node_class_mappings)
                except Exception as e:
                    logging.warning(f"Failed to load custom node from {filename}: {e}")
                    traceback.print_exc()
    return node_classes

# Function to determine if input type is a UI element
def is_ui_element(input_type):
    if isinstance(input_type, (tuple, list)):
        # Check if first element is a list of strings (dropdown options)
        if isinstance(input_type[0], list) and all(isinstance(opt, str) for opt in input_type[0]):
            return True
        elif isinstance(input_type[0], str):
            type_name = input_type[0]
            if type_name in UI_TYPES:
                return True
            # elif len(input_type) > 1:
            #     # Check if second element is a dict or list (parameters or dropdown options)
            #     if isinstance(input_type[1], dict):
            #         return True
            #     elif isinstance(input_type[1], (tuple, list)):
            #         return True  # Dropdown options
        # else:
        #     # If first element is not a str or list, assume UI element
        #     return True
    elif isinstance(input_type, list) and all(isinstance(opt, str) for opt in input_type):
        # Input type is a list of strings, assume dropdown
        return True
    elif isinstance(input_type, str):
        if input_type in UI_TYPES:
            return True
    return False

# Function to determine if input type is a data port
def is_data_port(input_type):
    if isinstance(input_type, (tuple, list)):
        if isinstance(input_type[0], str):
            type_name = input_type[0]
            if type_name in DATA_PORT_TYPES:
                return True
        elif isinstance(input_type[0], list):
            # If first element is a list (e.g., dropdown options), it's not a data port
            return False
    elif isinstance(input_type, str):
        if input_type in DATA_PORT_TYPES:
            return True
    return False

# Function to parse node inputs
def parse_node_inputs(node_class):
    ui_inputs = []
    port_inputs = []
    input_types = node_class.INPUT_TYPES()
    for key in ['required', 'optional', 'hidden']:
        if key in input_types:
            inputs_dict = input_types[key]
            for input_name, input_type in inputs_dict.items():
                if is_ui_element(input_type):
                    ui_inputs.append((input_name, input_type))
                else:
                    # Treat as data port
                    if isinstance(input_type, (tuple, list)):
                        if isinstance(input_type[0], str):
                            type_name = input_type[0]
                        else:
                            # Unknown type, consider as data port
                            type_name = input_name.upper()
                    else:
                        type_name = input_type
                    # Add to DATA_PORT_TYPES if not already present
                    if type_name not in DATA_PORT_TYPES:
                        DATA_PORT_TYPES.add(type_name)
                    port_inputs.append((input_name, input_type))
    return ui_inputs, port_inputs

# Function to parse node outputs
def parse_node_outputs(node_class):
    outputs = node_class.RETURN_TYPES
    if isinstance(outputs, (tuple, list)):
        outputs = list(outputs)
    else:
        outputs = [outputs]
    # Add output types to DATA_PORT_TYPES if not UI types
    for output_type in outputs:
        if output_type not in UI_TYPES and output_type not in DATA_PORT_TYPES:
            DATA_PORT_TYPES.add(output_type)
    return outputs

# Function to determine socket type based on data type
def determine_socket_type(data_type):
    socket_type_mapping = {
        'LATENT': 2,
        'IMAGE': 5,
        'MASK': 5,
        'CONDITIONING': 3,
        'VAE': 4,
        'CLIP': 4,
        'MODEL': 4,
        'CONTROL_NET': 4,
        # Add more mappings as necessary
        # Default socket type
        'DEFAULT': 7,
    }
    if isinstance(data_type, (tuple, list)):
        if isinstance(data_type[0], str):
            type_name = data_type[0]
        else:
            type_name = 'DEFAULT'
    else:
        type_name = data_type
    return socket_type_mapping.get(type_name, socket_type_mapping['DEFAULT'])

# Function to create a node
def create_node(node_class, node_name, ui_inputs, port_inputs, outputs, category_input, fn=None):
    max_width = 400
    total_height = 180

    class_name = node_name.replace(" ", "")
    class_code = 0

    # Create new Widget class
    class Widget(QDMNodeContentWidget):
        finished = Signal()

        def initUI(self):
            for item in ui_inputs:
                input_name = item[0]
                input_type = item[1]
                self.create_widget(input_name, input_type)
            self.create_main_layout(grid=1)

        def create_widget(self, widget_name, input_type):
            nonlocal max_width, total_height
            if isinstance(input_type, (tuple, list)):
                if isinstance(input_type[0], list) and all(isinstance(opt, str) for opt in input_type[0]):
                    # Dropdown options
                    options = input_type[0]
                    params = input_type[1] if len(input_type) > 1 else {}
                    widget = self.create_combo_box(widget_name, options, accessible_name=widget_name)
                elif isinstance(input_type[0], str):
                    type_name = input_type[0]
                    params = input_type[1] if len(input_type) > 1 else {}
                    widget = None
                    if type_name == 'INT':
                        min_val = params.get('min', default_numeric['INT']['min'])
                        max_val = params.get('max', default_numeric['INT']['max'])
                        default_val = params.get('default', default_numeric['INT']['default'])
                        step = params.get('step', default_numeric['INT']['step'])
                        widget = self.create_spin_box(label_text=widget_name, min_val=max(min_val, -2147483647), max_val=min(max_val, 2147483646), step=step, default_val=default_val)
                    elif type_name == 'FLOAT':
                        min_val = params.get('min', default_numeric['FLOAT']['min'])
                        max_val = params.get('max', default_numeric['FLOAT']['max'])
                        default_val = params.get('default', default_numeric['FLOAT']['default'])
                        step = params.get('step', default_numeric['FLOAT']['step'])
                        widget = self.create_double_spin_box(label_text=widget_name, min_val=min_val, max_val=max_val, step=step, default_val=default_val)
                    elif type_name in ['STRING', 'TEXT', 'PROMPT']:
                        default = params.get('default', '')
                        multiline = params.get('multiline', False)
                        if multiline:
                            widget = self.create_text_edit(widget_name, default=str(default))
                        else:
                            widget = self.create_line_edit(widget_name, default=str(default), placeholder=str(default))
                    elif type_name == 'BOOLEAN':
                        self.create_check_box(widget_name, spawn=widget_name.lower())
                    elif isinstance(input_type[1], (list, tuple)) and all(isinstance(opt, str) for opt in input_type[1]):
                        # Dropdown options
                        options = input_type[1]
                        widget = self.create_combo_box(widget_name, options, accessible_name=widget_name)
                    else:
                        # Default to line edit
                        default = params.get('default', '')
                        widget = self.create_line_edit(widget_name, default=str(default), placeholder=str(default))
                else:
                    # Default to line edit
                    default = ''
                    widget = self.create_line_edit(widget_name, default=str(default), placeholder=str(default))
            elif isinstance(input_type, list) and all(isinstance(opt, str) for opt in input_type):
                # Input type is a list of strings, assume dropdown
                options = input_type
                widget = self.create_combo_box(widget_name, options, accessible_name=widget_name)
            else:
                # Default to line edit
                default = ''
                widget = self.create_line_edit(widget_name, default=str(default), placeholder=str(default))
            if widget:
                setattr(self, widget_name, widget)
                # Connect signals to mark node dirty
                if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                    widget.valueChanged.connect(self.mark_node_dirty)
                elif isinstance(widget, (QLineEdit, QTextEdit)):
                    widget.textChanged.connect(self.mark_node_dirty)
                elif isinstance(widget, QComboBox):
                    widget.currentIndexChanged.connect(self.mark_node_dirty)
                # Update total height and max width
                if hasattr(widget, 'sizeHint'):
                    widget_size = widget.sizeHint()
                    total_height += widget_size.height()
                    max_width = max(max_width, widget_size.width())
                else:
                    total_height += 30

        @QtCore.Slot()
        def mark_node_dirty(self, value=None):
            self.node.markDirty(True)

    # Create new Node class
    class Node(AiNode):

        icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
        help_text = "Generated node from ComfyUI"
        op_code = class_code
        op_title = node_name
        content_label_objname = class_name.lower().replace(" ", "_")
        category = f"{category_input}/{getattr(node_class, 'CATEGORY', 'Default')}"
        NodeContent_class = Widget
        dim = (max_width, total_height)
        output_data_ports = outputs
        exec_port = len(outputs)

        custom_input_socket_names = [name for name, _ in port_inputs] + ['EXEC']
        custom_output_socket_names = outputs + ['EXEC']

        make_dirty = False if 'loader' in node_name.lower() else True

        def __init__(self, scene):
            inputs = [determine_socket_type(t) for _, t in port_inputs] + [1]
            outputs_sockets = [determine_socket_type(t) for t in outputs] + [1]
            super().__init__(scene, inputs=inputs, outputs=outputs_sockets)
            self.fn = fn if fn else getattr(node_class, getattr(node_class, 'FUNCTION', None), None)
            self.node_class = node_class  # Store node_class without instantiating
            self.output_names = outputs
            self.output_types = outputs_sockets
            self.exec_port = len(self.outputs) - 1

            modifier = len(inputs)
            if len(outputs_sockets) > len(inputs):
                modifier = len(outputs_sockets)

            self.content.setGeometry(0, 15, self.content.geometry().width(), self.content.geometry().height())

            self.grNode.height = self.content.geometry().height() + (modifier * 30)

            self.update_all_sockets()

            self.ui_inputs = ui_inputs
            self.port_inputs = port_inputs
            self.adapted_outputs = outputs

            self.cache = {}

            self.device = 'cuda'
            self.loaded_lora = None

        @torch.inference_mode()
        def evalImplementation_thread(self):
            data_inputs = {}
            # Get data from port inputs
            for idx, (input_name, _) in enumerate(self.port_inputs):
                data = self.getInputData(idx)
                if data is not None:
                    data_inputs[input_name] = data

            # Get data from UI inputs
            for ui_input in self.ui_inputs:
                input_name = ui_input[0]
                input_type = ui_input[1]
                widget = getattr(self.content, input_name)
                if isinstance(input_type, (tuple, list)):
                    if isinstance(input_type[0], list) and all(isinstance(opt, str) for opt in input_type[0]):
                        # Dropdown
                        data = widget.currentText()
                        data_inputs[input_name] = data
                    else:
                        type_name = input_type[0]
                        if type_name in ['INT', 'FLOAT']:
                            data = widget.value()
                            data_inputs[input_name] = data
                        elif type_name in ['STRING', 'TEXT', 'PROMPT', 'BOOLEAN']:
                            if isinstance(widget, QTextEdit):
                                data = widget.toPlainText()
                            elif isinstance(widget, QLineEdit):
                                data = widget.text()
                            else:
                                data = widget.text()
                            data_inputs[input_name] = data
                        elif isinstance(input_type[1], (list, tuple)) and all(isinstance(opt, str) for opt in input_type[1]):
                            # Dropdown with options in second element
                            data = widget.currentText()
                            data_inputs[input_name] = data
                        else:
                            # Default to text
                            data = widget.text()
                            data_inputs[input_name] = data
                elif isinstance(input_type, list) and all(isinstance(opt, str) for opt in input_type):
                    # Dropdown
                    data = widget.currentText()
                    data_inputs[input_name] = data
                else:
                    # Default to text
                    if isinstance(widget, QTextEdit):
                        data = widget.toPlainText()
                    elif isinstance(widget, QLineEdit):
                        data = widget.text()
                    else:
                        data = widget.text()
                    data_inputs[input_name] = data

            # Additional inputs from connected nodes
            all_ins = self.getAllInputs()
            if "exec" in all_ins:
                del all_ins["exec"]
            data_inputs.update(all_ins)

            if "seed" in data_inputs:
                if data_inputs.get("seed", 0) == 0:
                    data_inputs["seed"] = secrets.randbelow(2147483647)
                    self.markDirty(True)

            if self.make_dirty:
                self.markDirty(True)
            if self.isDirty():
                with torch.no_grad():
                    # Instantiate the node_class only when needed
                    node_instance = self.node_class()
                    result = self.fn(node_instance, **data_inputs)

                # Set outputs
                for idx, output_name in enumerate(self.adapted_outputs):
                    if output_name != "EXEC":
                        self.setOutput(idx, result[idx])
                self.markDirty(False)
            self.content.update()
            self.content.finished.emit()
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
                        return False

            return True

        def onInputChanged(self, socket=None):
            self.markDirty(True)

        def remove(self):
            # Clean up
            for index in range(len(self.outputs)):
                object_name = self.getID(index)
                if hasattr(self, object_name):
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
                    delattr(self, object_name)

            self.cache.clear()
            super().remove()

    register_node_now(Node)

# Main code to load nodes and create them
def main():
    # Set COMFYUI_PATH to the path where ComfyUI is installed
    COMFYUI_PATH = 'src/ComfyUI'  # Replace with the actual path to ComfyUI

    # Add ComfyUI path to sys.path
    if COMFYUI_PATH not in sys.path:
        sys.path.append(COMFYUI_PATH)

    # Import ComfyUI base nodes
    try:
        import nodes
    except ImportError as e:
        logging.error(f"Failed to import ComfyUI base nodes: {e}")
        sys.exit(1)
    nodes.init_extra_nodes(init_custom_nodes=True)

    # Initialize node class mappings with base nodes
    NODE_CLASS_MAPPINGS = nodes.NODE_CLASS_MAPPINGS.copy()

    # Now, for each node, parse inputs and outputs and create the node
    for node_name, node_class in NODE_CLASS_MAPPINGS.items():
        try:
            ui_inputs, port_inputs = parse_node_inputs(node_class)
            outputs = parse_node_outputs(node_class)
            category = getattr(node_class, 'CATEGORY', 'Default')
            fn = getattr(node_class, getattr(node_class, 'FUNCTION', None), None)
            create_node(node_class, node_name, ui_inputs, port_inputs, outputs, category_input="ComfyUI Nodes", fn=fn)
        except Exception as e:
            logging.warning(f"Failed to create node {node_name}: {e}")
            traceback.print_exc()

# Run the main function
main()
