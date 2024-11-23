import inspect
import os

# import cv2
import numpy as np
import torch
from PIL import Image
# from PIL.ImageQt import ImageQt
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QUrl, pyqtSignal, QMimeData, QBuffer
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QLabel, QFileDialog


from node_core.corenode import CalcNode, CalcGraphicsNode, CalcContent
from ainodes_core.except_handler import handle_ainodes_exception
from node_core.worker import Worker
from nodeeditor.node_content_widget import QDMNodeContentWidget
from nodeeditor.node_node import Node
from nodeeditor.node_serializable import Serializable
from nodeeditor.node_socket import LEFT_BOTTOM, RIGHT_BOTTOM

LISTBOX_MIMETYPE = "application/x-item"

# Define an empty dictionary to store node classes
NODE_CLASSES = {}
NODE_REGISTRY = {}


from enum import Enum

# Enum for different data types
class DataType(Enum):
    STRING = "string"
    NUMBER = "number"
    LIST = "list"
    IMAGE = "image"
    TENSOR = "tensor"
    DICT = "dict"
    UNKNOWN = "unknown"

# Dictionary to map data types to their corresponding control widgets or socket types
IO_MAPPING = {
    DataType.STRING: "StringControlWidget",   # Example widget for string inputs
    DataType.NUMBER: "NumberControlWidget",   # Example widget for number inputs
    DataType.LIST: "ListControlWidget",       # Example widget for list inputs
    DataType.IMAGE: "ImageSocket",            # Example socket for image inputs
    DataType.TENSOR: "TensorSocket",          # Example socket for tensor inputs
    DataType.DICT: "DictSocket",              # Example socket for dict inputs
    DataType.UNKNOWN: "UnknownSocket",        # Fallback socket for unknown types
}

# Dictionary to map data types to control widgets for UIs
WIDGET_MAPPING = {
    DataType.STRING: "StringWidget",   # Widget for string data
    DataType.NUMBER: "NumberWidget",   # Widget for number data
    DataType.LIST: "ListWidget",       # Widget for list data
    DataType.IMAGE: "ImageWidget",     # Widget for image data
    DataType.TENSOR: "TensorWidget",   # Widget for tensor data
    DataType.DICT: "DictWidget",       # Widget for dict data
    DataType.UNKNOWN: "DefaultWidget", # Fallback widget for unknown types
}

# Custom exceptions
class ConfException(Exception): pass
class InvalidNodeRegistration(ConfException): pass
class NodeNotRegistered(ConfException): pass


import re
class AiNode(Node):
    icon = ""
    _output_values = {}  # A dictionary to store references to output values
    thumbnail = None
    op_code = 0
    op_title = "Undefined"
    content_label = ""

    category = "default"
    input_socket_name = ["EXEC"]
    output_socket_name = ["EXEC"]
    help_text = "Default help text"
    GraphicsNode_class = CalcGraphicsNode
    NodeContent_class = CalcContent
    sockets = None
    use_gpu = False

    def __init__(self, scene, inputs=[2, 2], outputs=[1]):
        # self.threadpool = QThreadPool()
        """
         Initialize the AiNode class with a scene, inputs, and outputs.

         Args:
             scene (QGraphicsScene): The scene where the node is displayed.
             inputs (list): A list of input sockets, with values representing their types.
                           1: EXEC
                           2: LATENT
                           3: CONDITIONING
                           4: PIPE/MODEL
                           5: IMAGE
                           6: DATA
                           7: STRING
                           8: INT
                           9: FLOAT
                           Defaults to [2,2].
             outputs (list): A list of output sockets, with values representing their types.
                           1: EXEC
                           2: LATENT
                           3: CONDITIONING
                           4: PIPE/MODEL
                           5: IMAGE
                           6: DATA
                           7: STRING
                           8: INT
                           9: FLOAT
                           Defaults to [1].
         """
        super().__init__(scene, self.__class__.op_title, inputs, outputs)
        #self.content_label_objname = self.__class__.__name__.lower().replace(" ", "_")
        self.set_socket_names()
        self.value = None
        self.output_values = {}
        # it's really important to mark all nodes Dirty by default
        self.markDirty()
        self.values = {}
        self.busy = False
        self.init_done = None
        self.exec_port = len(self.outputs) - 1
        self.worker = None

    def initInnerClasses(self):
        self._output_values = {}
        node_content_class = self.getNodeContentClass()
        graphics_node_class = self.getGraphicsNodeClass()
        if node_content_class is not None: self.content = node_content_class(self)
        if graphics_node_class is not None: self.grNode = graphics_node_class(self)
        if hasattr(self, "dim"):
            width = self.dim[0]
            height = self.dim[1]
            self.grNode.width = width
            self.grNode.height = height
            self.content.setMinimumHeight(height - 125)
            self.content.setMinimumWidth(width - 10)
        self.grNode.icon = QtGui.QImage(self.icon)
        self.grNode.thumbnail = self.grNode.icon.scaled(64, 64, QtCore.Qt.KeepAspectRatio)
        self.content.eval_signal.connect(self.evalImplementation)

    def set_socket_names(self):
        """
        Internal function to set socket names, override in your custom node pack to add additional socket types
        """

        if self.sockets == None:
            sockets = {1: "EXEC",
                       2: "LATENT",
                       3: "COND",
                       4: "PIPE/MODEL",
                       5: "IMAGE",
                       6: "DATA",
                       7: "STRING",
                       8: "INT",
                       9: "FLOAT", }
        else:
            sockets = self.sockets

        # Initialize the input_socket_name and output_socket_name lists as empty lists
        self.input_socket_name = []
        self.output_socket_name = []

        # Dynamically populate input and output socket names using loops
        for input_index in self._inputs:
            self.input_socket_name.append(sockets[input_index])
        for output_index in self._outputs:
            self.output_socket_name.append(sockets[output_index])

        if hasattr(self, "custom_input_socket_names"):
            self.input_socket_name = self.custom_input_socket_names
        if hasattr(self, "custom_output_socket_names"):
            self.output_socket_name = self.custom_output_socket_names

        self.initSockets(inputs=self._inputs, outputs=self._outputs, reset=True)

    def update_all_sockets(self):
        for socket in self.outputs + self.inputs:
            socket.setSocketPosition()
        self.updateConnectedEdges()

    def get(self):
        res = self.content.serialize()
        return res

    def getID(self, index):
        """
        Generate a unique ID for the output socket with the given index.

        Args:
            index (int): The index of the output socket.

        Returns:
            str: The unique ID for the output socket.
        """
        return f"{id(self)}_output_{index}"

    def setOutput(self, index, value):
        """
        Set the value of the output socket with the given index.

        Args:
            index (int): The index of the output socket.
            value: The value to be set for the output socket.
        """
        object_name = self.getID(index)
        self._output_values[object_name] = value  # Store the reference in the dictionary

    def getOutput(self, index=0, origin_index=0):
        """
         Get the value of the output socket with the given index.

         Args:
             index (int): The index of the output socket.

         Returns:
             The value of the output socket, or None if it does not exist.
         """
        object_name = self.getID(index)
        return self._output_values.get(object_name, None)  # Get the value using the dictionary

    def getInputData(self, index=0, origin_index=0):
        """
        Retrieve data from a connected input socket.

        Sockets are numbered from the bottom of the node, starting at 0.
        For example, in a node with 3 inputs, the bottom-most socket is 0, and the top-most is 2.

        :param index: The index of the input socket to get data from.
        :type index: int
        :return: The data from the connected input socket, or None if the socket is not connected.
        :rtype: Any or None
        """

        try:
            if len(self.getInputs(index)) > 0:
                node, new_index = self.getInput(index)
                # print("node", node)
                return node.getOutput(new_index, index)
            else:
                return None
        except Exception as e:
            #done = handle_ainodes_exception()
            print(f"Error in getInputData: {e}")
            return None

    def getAllInputs(self):
        ser_content = self.content.serialize(exec=True) if isinstance(self.content, Serializable) else {}
        for input in self.inputs:
            ser_content[input.name.lower()] = self.getInputData(input.index)
        return ser_content

    def initSettings(self):
        """
        Initialize settings for the node, such as input and output socket positions.
        """
        super().initSettings()
        self.input_socket_position = LEFT_BOTTOM
        self.output_socket_position = RIGHT_BOTTOM

    def finishInitialization(self):
        if not self.content.eval_signal.hasConnections():
            self.content.eval_signal.connect(self.evalImplementation)
            self.init_done = True

    def evalOperation(self, input1, input2):
        """
        Evaluate the operation using the given inputs.

        Args:
            input1: The first input value.
            input2: The second input value.

        Returns:
            The result of the operation.
        """
        return 123

    ##@QtCore.Slot()
    def evalImplementation(self, index=0, *args, **kwargs):

        #if gs.should_run:
        if not self.busy:

            self.busy = True
            self.worker = Worker(self.evalImplementationThreadHandler)
            self.worker.signals.result.connect(self.onWorkerFinished)
            # self.worker.signals.interrupt.connect(self.stop_worker)
            self.scene.threadpool.start(self.worker)
            return
        else:
            return
        # else:
        #     return

    def evalImplementationThreadHandler(self, *args, **kwargs):
        # with torch.inference_mode():
        #     result = self.evalImplementation_thread()
        #     return result
        try:

            result = self.evalImplementation_thread()
            return result
        except:
            self.busy = False
            handle_ainodes_exception()
            return None
        #     #
        #     return []

    def stop_worker(self):
        self.worker.stop_fn()

    def evalImplementation_thread(self):
        """
        The core evaluation logic of the node, to be run in a separate thread.
        This method should implement the processing logic of the node,
        which is executed when the node is evaluated.
        """
        return None

    def clearOutputs(self):
        ports = list(range(len(self.outputs) - 1))
        for port in ports:
            self.setOutput(port, None)

    def onWorkerFinished(self, result, exec=True):
        """
        Callback method that is called when the worker thread has finished processing.

        The 'result' argument should always be a list, as this method sets the node's outputs based on this list.
        Each element in the list corresponds to an output socket, in the order of the sockets.

        :param result: The result data from the worker thread, expected to be a list.
        :param exec: Flag to indicate if subsequent connected nodes should be triggered for execution.
        """
        self.busy = False
        if result is not None and (isinstance(result, list) or isinstance(result, tuple)):

            if hasattr(self, "output_data_ports"):
                ports = self.output_data_ports

            else:
                ports = list(range(len(self.outputs) - 1))

            x = 0
            for port in ports:
                # print(len(result), x)
                if len(result) - 1 >= x:

                    if result[x] is not None:
                        self.markDirty(False)
                    # print(f"setting port {port} to {result[x]}")
                    self.setOutput(port, result[x])
                x += 1

        if exec:
            if hasattr(self, "exec_port"):
                port = self.exec_port
            else:
                port = len(self.outputs) - 1
            if len(self.getOutputs(port)) > 0:
                self.executeChild(output_index=port)

    def eval(self, index=0):
        # try:
        self.content.eval_signal.emit()
        # except Exception as e:
        #     print(e, self)

    def eval_orig(self, index=0):
        """
        Evaluate the node, returning the cached value if it's valid and not dirty.

        Args:
            index (int): Optional index used for custom implementations. Defaults to 0.

        Returns:
            The result of the operation, or None if the evaluation failed.
        """
        if not self.isDirty() and not self.isInvalid():
            print(" _> returning cached %s value:" % self.__class__.__name__, self.value)
            return self.value
        try:
            self.evalImplementation(index)
            return None
        except ValueError as e:
            self.markInvalid()
            self.grNode.setToolTip(str(e))
            # self.markDescendantsDirty()
        except Exception as e:
            self.markInvalid()
            self.grNode.setToolTip(str(e))
            #dumpException(e)

    def executeChild(self, output_index=0):
        """
        Execute the child node connected to the output socket with the given index.

        Args:
            output_index (int): The index of the output socket. Defaults to 0.

        Returns:
            None
        """
        if len(self.getOutputs(output_index)) > 0:
            try:
                node = self.getOutputs(output_index)[0]
                # node.markDirty(True)
                node.eval()
            except Exception as e:
                print("Skipping execution:", e, self)

    def onInputChanged(self, socket=None):
        """
        Handle the event when an input socket value is changed.

        Args:
            socket: The input socket that changed.
        """
        print("%s::__onInputChanged" % self.__class__.__name__)
        # self.markDirty(True)

    def update_vars(self, data):

        if data is not None:
            try:
                for key, value in data.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
                return None
            except:
                #handle_ainodes_exception()
                return None

    def serialize(self):
        """
        Serialize the node's data into a dictionary.

        Returns:
            dict: The serialized data of the node.
        """
        res = super().serialize()
        res['op_code'] = self.__class__.op_code
        res['content_label_objname'] = self.content_label_objname
        return res

    def deserialize(self, data, hashmap={}, restore_id=True):
        """
        Deserialize the node's data from a dictionary.

        Args:
            data (dict): The serialized data of the node.
            hashmap (dict): A dictionary of node IDs and their corresponding objects.
            restore_id (bool): Whether to restore the original node ID. Defaults to True.

        Returns:
            bool: True if deserialization is successful, False otherwise.
        """
        res = super().deserialize(data, hashmap, restore_id)
        # print("Deserialized AiNode '%s'" % self.__class__.__name__, "res:", res)
        return res

    def remove(self):
        """
         Remove the node, clearing the values in its output sockets.
         """
        x = 0
        self.clearOutputs()
        self.values = None
        self.values = {}
        super().remove()

    def showNiceDialog(self):
        title = self.title + " Help"
        dialog = QtWidgets.QDialog(self.scene.getView())
        dialog.setWindowTitle(title)
        dialog.setWindowFlags(dialog.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(QLabel(self.help_text))

        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(dialog.close)
        layout.addWidget(close_button)

        dialog.setLayout(layout)
        dialog.show()
        return dialog

    def can_run(self):

        if hasattr(self, "force_run"):
            if self.force_run:
                return True
        for socket in self.inputs:
            if socket.is_input and socket.hasAnyEdge():
                for edge in socket.edges:
                    if edge.start_socket.node is not self:
                        if edge.start_socket.node.isDirty():
                            return False
                    if edge.end_socket.node is not self:
                        if edge.end_socket.node.isDirty():
                            return False
        return True

def parse_docstring(docstring, params):
    param_metadata = {}

    if docstring:
        # Split docstring by parameters
        param_blocks = docstring.split('- ')[1:]  # Skip the first element which is before the first '- '

        for block in param_blocks:
            lines = block.split('\n')
            param_line = lines[0].strip()
            param_name = param_line.split(':')[0].strip()

            # Determine the type of the parameter
            param_type = params.get(param_name, {}).get('type', None)

            metadata = {
                "name": None,
                "min": None,
                "max": None,
                "step": None,
                "default": None,
                "type": param_type,
                "options": None,
                "options_func": None
            }

            # Function to convert string to appropriate type
            def convert_value(value):
                if param_type == int:
                    return int(value)
                elif param_type == float:
                    return float(value)
                elif param_type == str:
                    return value
                elif param_type == bool:
                    return value.lower() == 'true'
                else:
                    try:
                        return eval(value)  # Try to evaluate to a number if possible
                    except:
                        return value  # Default to string if evaluation fails

            # Search for metadata in the block
            name_match = re.search(r"name\s*=\s*(-?\d+\.?\d*)", block)
            min_match = re.search(r"min\s*=\s*(-?\d+\.?\d*)", block)
            max_match = re.search(r"max\s*=\s*(-?\d+\.?\d*)", block)
            step_match = re.search(r"step\s*=\s*(-?\d+\.?\d*)", block)
            default_match = re.search(r"default\s*=\s*(-?\d+\.?\d*|True|False)", block)
            options_match = re.search(r"options\s*=\s*\[(.*?)\]", block)
            options_func_match = re.search(r"options_func\s*=\s*(\w+)", block)

            if name_match:
                metadata["name"] = str(min_match.group(1))
            if min_match:
                metadata["min"] = convert_value(min_match.group(1))
            if max_match:
                metadata["max"] = convert_value(max_match.group(1))
            if step_match:
                metadata["step"] = convert_value(step_match.group(1))
            if default_match:
                metadata["default"] = convert_value(default_match.group(1))
            if options_match:
                options = options_match.group(1).split(',')
                metadata["options"] = [option.strip().strip('"').strip("'") for option in options]
                metadata["type"] = list  # Indicate that this is a list of options
            if options_func_match:
                metadata["options_func"] = options_func_match.group(1)
                metadata["type"] = 'function'  # Indicate that this is a function for generating options

            param_metadata[param_name] = metadata

    return param_metadata


def register_node_now(class_reference, name=None):
    class_name = class_reference.content_label_objname

    if class_name in NODE_CLASSES:
        raise InvalidNodeRegistration(f"Duplicate node registration of '{class_name}'. There is already {NODE_CLASSES[class_name]}")
    if name and name in NODE_CLASSES:
        raise InvalidNodeRegistration(
            f"Duplicate node registration of '{name}'. There is already {NODE_CLASSES[name]}")
    if name:
        NODE_CLASSES[name] = class_reference
    else:
        NODE_CLASSES[class_name] = class_reference



def register_node():
    def decorator(original_class):
        isclass = False
        if inspect.isclass(original_class):
            isclass = True
            # For classes, inspect the __init__ and __call__ methods
            init_signature = inspect.signature(original_class.__init__)
            call_signature = inspect.signature(original_class.__call__)

            # Collect parameter info from __init__
            init_params = {}
            for name, param in init_signature.parameters.items():
                if name != 'self':
                    init_params[name] = {
                        'type': param.annotation,
                        'default': param.default if param.default is not inspect.Parameter.empty else None
                    }

            # Collect parameter info from __call__
            call_params = {}
            for name, param in call_signature.parameters.items():
                if name != 'self':
                    call_params[name] = {
                        'type': param.annotation,
                        'default': param.default if param.default is not inspect.Parameter.empty else None
                    }

            # Register the class with its collected information
            NODE_REGISTRY[original_class.__name__] = {
                'type': 'class',
                'class': original_class,
                'init_params': init_params,
                'call_params': call_params
            }

        elif inspect.isfunction(original_class):
            # For functions, inspect the function's signature
            func_signature = inspect.signature(original_class)

            # Collect parameter info from the function
            func_params = {}
            for name, param in func_signature.parameters.items():
                func_params[name] = {
                    'type': param.annotation,
                    'default': param.default if param.default is not inspect.Parameter.empty else None
                }
            # Determine the return type for function
            return_type = func_signature.return_annotation
            if return_type is inspect.Signature.empty:
                return_type = None
            docstring = original_class.__doc__
            param_metadata = parse_docstring(docstring, func_params)  # Pass func_params to include type info
            # Register the function with its collected information
            NODE_REGISTRY[original_class.__name__] = {
                'type': 'function',
                'function': original_class,
                'params': func_params,
                'metadata': param_metadata,  # Store the additional metadata
                'return_type': return_type
            }
        if isinstance(original_class, type):
            if issubclass(original_class, Node):
                register_node_now(original_class)
        else:
            node_class = parse_node(NODE_REGISTRY[original_class.__name__], isclass)
            register_node_now(node_class, name=original_class.__name__)
        return original_class
    return decorator



def parse_node(node_data, isclass):
    """
    This function parses node data and generates a new node class with appropriate widgets and inputs.
    """
    # Define the Node Content Widget
    class NodeContentWidget(QDMNodeContentWidget):
        def initUI(self):
            self.widgets = {}
            # Create widgets based on the function's metadata
            # if 'metadata' in node_data:
            #     for param_name, meta in node_data['metadata'].items():
            #         # If min and max are defined, assume it's a numeric widget
            #         if meta['min'] is not None and meta['max'] is not None:
            #             self.create_double_spin_box(param_name, meta['min'], meta['max'], meta['default'], meta['step'])
            # self.create_main_layout(grid=True)
            if 'metadata' in node_data:
                for param_name, meta in node_data['metadata'].items():
                    param_type = meta.get('type')
                    if param_type == str:
                        self.create_text_edit(param_name, meta.get('default', ''))
                    elif param_type == int:
                        self.create_spin_box(param_name, meta.get('min', 0), meta.get('max', 100), meta.get('default', 0), meta.get('step', 1))
                    elif param_type == float:
                        self.create_double_spin_box(param_name, meta.get('min', 0.0), meta.get('max', 1.0), meta.get('default', 0.0), meta.get('step', 0.1))
                    elif param_type == bool:
                        self.create_check_box(param_name, meta.get('default', False))
                    elif param_type == list and meta.get('options') is not None:
                        self.create_combo_box(param_name, meta['options'], meta.get('default', None))
                    elif param_type == 'function' and meta.get('options_func') is not None:
                        options = globals()[meta['options_func']]()
                        self.create_combo_box(param_name, options, meta.get('default', None))
            self.create_main_layout(grid=True)

    # Define the Node class
    class ParsedNode(AiNode):
        icon = ""
        op_code = 0
        op_title = node_data['function'].__name__
        content_label = f"{op_title} Content"

        dim = (340, 180)
        GraphicsNode_class = CalcGraphicsNode
        NodeContent_class = NodeContentWidget

        def __init__(self, scene, inputs=None, outputs=None):
            if inputs is None:
                inputs = []
            if outputs is None:
                outputs = []

            self.custom_input_socket_names = []
            self.custom_output_socket_names = []

            # Determine inputs based on the function's signature and annotations
            for param_name, param_info in node_data['params'].items():
                param_type = param_info['type']
                if param_type == Image.Image:
                    inputs.append((2))  # Assume (2, 2) as placeholder values
                    self.custom_input_socket_names.append(param_name)
                else:
                    inputs.append(9)
                    self.custom_input_socket_names.append(param_name)


            # Determine if an output of type IMAGE should be added
            return_type = node_data.get('return_type')
            if return_type == Image.Image:
                outputs.append(2)  # Add output for IMAGE type
                self.custom_output_socket_names.append('image')
            else:
                outputs.append(9)
                self.custom_output_socket_names.append('generator')

            inputs.append(1)
            outputs.append(1)
            self.custom_input_socket_names.append('exec')
            self.custom_output_socket_names.append('exec')

            super().__init__(scene, self.op_title, inputs, outputs)
            self.content_label_objname = self.op_title
            self.named_params = node_data['params'].keys()
            if hasattr(self, "dim"):
                width = self.dim[0]
                height = self.dim[1]
                self.grNode.width = width
                self.grNode.height = height
                self.content.setMinimumHeight(height - 125)
                self.content.setMinimumWidth(width - 10)
            self.content.setGeometry(25, 50, self.content.geometry().width(), self.content.geometry().height())


            self.grNode.height = self.content.geometry().height() + (5 * 30)
            self.grNode.width = self.content.geometry().width() + 50

            self.update_all_sockets()

        def evalImplementation_thread(self, **kwargs):
            # Extract the original function's parameters from the node data
            original_function = node_data['function']
            func_signature = inspect.signature(original_function)
            func_params = func_signature.parameters

            # Retrieve all input data
            dynamic_inputs = self.getAllInputs()

            # Prepare the arguments to call the original function
            call_args = []
            call_kwargs = {}

            for param_name, param in func_params.items():
                if param.kind in [inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD]:
                    # Add positional arguments
                    if param_name in dynamic_inputs:
                        call_args.append(dynamic_inputs[param_name])
                    elif param.default is not inspect.Parameter.empty:
                        # Use default value if not provided
                        call_args.append(param.default)
                    else:
                        raise ValueError(f"Missing required positional argument: '{param_name}'")

                elif param.kind == inspect.Parameter.KEYWORD_ONLY:
                    # Add keyword-only arguments
                    if param_name in dynamic_inputs:
                        call_kwargs[param_name] = dynamic_inputs[param_name]
                    elif param.default is not inspect.Parameter.empty:
                        call_kwargs[param_name] = param.default
                    else:
                        raise ValueError(f"Missing required keyword argument: '{param_name}'")

            # Call the original function with the constructed arguments
            # try:
            result = original_function(*call_args, **call_kwargs)
            # except TypeError as e:
            #     print(f"Error calling function {original_function.__name__}: {e}")
            #     result = None

            return [result]


    return ParsedNode

def tensor_image_to_pixmap(tensor_image):
    """ Convert a tensor or NumPy array image to a QPixmap. """
    if isinstance(tensor_image, torch.Tensor):
        pil_image = tensor2pil(tensor_image)
    elif isinstance(tensor_image, np.ndarray):
        pil_image = Image.fromarray(tensor_image)
    else:
        pil_image = tensor_image

    # Convert PIL Image to a QImage
    pil_image = pil_image.convert("RGBA")
    data = pil_image.tobytes("raw", "RGBA")
    qimage = QImage(data, pil_image.width, pil_image.height, QImage.Format_RGBA8888)

    # Convert QImage to QPixmap
    pixmap = QPixmap.fromImage(qimage)
    return pixmap

def tensor2pil(image):
    if image is not None:
        with torch.inference_mode():
            return Image.fromarray(np.clip(255. * image.detach().cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
    else:
        return None


# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


# generator = None
# def get_samplers():
#     # Return a list of available samplers
#     return modules.sampling.samplers.SAMPLER_NAMES
# def get_schedulers():
#     # Return a list of available samplers
#     return modules.sampling.samplers.SCHEDULER_NAMES
#
#
#
# @register_node()
# def get_flux(offload:bool=False, model:Flux=None, clip:CLIP=None, vae:VAE=None) -> FluxGenerator:
#     """
#     Flux Loader
#     Parameters:
#     - offload: bool, required (default=False)
#     default=False
#     """
#     print("NODE SETTING OFFLOAD TO", offload)
#     global generator
#     if generator == None:
#         generator = FluxGenerator(model=model, clip=clip, vae=vae)
#     # else:
#     #     generator.set_offload(offload)
#     return generator
#
# @register_node()
# def infer_flux(prompt:str="",
#                seed:int=-1,
#                steps:int=4,
#                width:int=512,
#                height:int=512,
#                sampler_name: str = "euler",
#                scheduler_name: str = "simple",
#                generator:FluxGenerator = None) -> Image.Image:
#
#     """
#     Flux inference.
#     Parameters:
#     - prompt: str, optional (default="")
#         The alpha blending value.
#         default=""
#     - seed: int, optional (default=-1)
#         The beta blending value.
#         min=-1, max=2147483646, step=1, default=-1
#     - steps: int, optional (default=4)
#         Steps
#         min=1, max=1000, step=1, default=4
#     - width: int, optional (default=512)
#         Width
#         min=8, max=4096, step=8, default=512
#     - height: int, optional (default=512)
#         Height
#         min=8, max=4096, step=8, default=512
#     - sampler_name: str, optional (default="euler")
#         The sampler to use for image generation.
#         options_func=get_samplers
#         default="euler"
#     - scheduler_name: str, optional (default="simple")
#         The scheduler to use for image generation.
#         options_func=get_schedulers
#         default="simple"
#     - generator: FluxGenerator, optional (default="simple")
#         The Generator
#
#     """
#     image = generator(prompt,
#                       steps=steps,
#                       width=width,
#                       height=height,
#                       save=False,
#                       seed=seed,
#                       sampler_name=sampler_name,
#                       scheduler_name=scheduler_name)
#     return image
#
# def encode_image(image):
#     torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def get_class_from_name(class_name):
    if class_name not in NODE_CLASSES:
        raise NodeNotRegistered(f"Node class '{class_name}' is not registered")
    return NODE_CLASSES[class_name]

# Import all nodes and register them
from base_nodes import *
# from examples.example_calculator.nodes. import *
# print(NODE_REGISTRY)
