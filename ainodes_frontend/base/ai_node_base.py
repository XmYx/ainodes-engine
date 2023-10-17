import gc
import threading

import torch
from qtpy import QtWidgets, QtCore, QtGui
from qtpy.QtCore import QRectF
from qtpy.QtGui import QImage
from qtpy.QtWidgets import QLabel

from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend.node_engine.node_graphics_node import QDMGraphicsNode
from ainodes_frontend.node_engine.node_node import Node
from ainodes_frontend.node_engine.node_socket import LEFT_BOTTOM, RIGHT_BOTTOM
from ainodes_frontend.node_engine.utils import dumpException
from .settings import handle_ainodes_exception
from .worker import Worker

from ainodes_frontend import singleton as gs



class CalcGraphicsNode(QDMGraphicsNode):
    icon = None
    thumbnail = None
    def initSizes(self):
        """
        Initialize the sizes and padding for the graphical representation of the node.
        """
        super().initSizes()
        self.width = 160
        self.height = 74
        self.edge_roundness = 6
        self.edge_padding = 0
        self.title_horizontal_padding = 8
        self.title_vertical_padding = 10


    def initAssets(self):
        """
        Initialize the assets (such as icons) for the graphical representation of the node.
        """
        super().initAssets()
        self.icons = QImage("ainodes_frontend/icons/status_icons.png")

    def paint(self, painter, QStyleOptionGraphicsItem, widget=None):
        """
        Paint the node on the QGraphicsView.

        Args:
            painter (QPainter): The painter used for drawing.
            QStyleOptionGraphicsItem (QStyleOptionGraphicsItem): Options for the graphical item being painted.
            widget (QWidget, optional): The widget being painted on, if any. Defaults to None.
        """

        super().paint(painter, QStyleOptionGraphicsItem, widget)

        offset = 24.0
        if self.node.isDirty(): offset = 0.0
        if self.node.isInvalid(): offset = 48.0

        painter.drawImage(
            QRectF(-10, -10, 24.0, 24.0),
            self.icons,
            QRectF(offset, 0, 24.0, 24.0)
        )

        # Paint self.icon at the top-right corner

        if self.thumbnail:
            icon_rect = QRectF(self.width - 40, -15, 48.0, 48.0)
            painter.drawImage(icon_rect, self.thumbnail)

class CalcContent(QDMNodeContentWidget):

    def initUI(self):
        lbl = QLabel(self.node.content_label, self)
        lbl.setObjectName(self.node.content_label_objname)


class WorkerThread(threading.Thread):
    def __init__(self, target, on_finished):
        super().__init__()
        self.target = target
        self.on_finished = on_finished

    def run(self):
        result = self.target()
        self.on_finished(result)

class AiNode(Node):
    icon = ""
    _output_values = {}  # A dictionary to store references to output values
    thumbnail = None
    op_code = 0
    op_title = "Undefined"
    content_label = ""
    content_label_objname = "calc_node_bg"
    category = "default"
    input_socket_name = ["EXEC"]
    output_socket_name = ["EXEC"]
    help_text = "Default help text"
    GraphicsNode_class = CalcGraphicsNode
    NodeContent_class = CalcContent
    sockets = None
    use_gpu = False

    def __init__(self, scene, inputs=[2,2], outputs=[1]):
        #self.threadpool = QThreadPool()
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
                           (4 is not used yet)
                           Defaults to [2,2].
             outputs (list): A list of output sockets, with values representing their types.
                           1: EXEC
                           2: LATENT
                           3: CONDITIONING
                           4: PIPE/MODEL
                           5: IMAGE
                           6: DATA
                           (4 is not used yet)
                           Defaults to [1].
         """
        super().__init__(scene, self.__class__.op_title, inputs, outputs)
        self.set_socket_names()
        self.value = None
        self.output_values = {}
        # it's really important to mark all nodes Dirty by default
        self.markDirty()
        self.values = {}
        self.busy = False
        self.init_done = None
        self.exec_port = len(self.outputs) - 1
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
            self.content.setMinimumHeight(height - 100)
            self.content.setMinimumWidth(width)
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
                       7: "STRING"}
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

        if hasattr(self, "custom_input_socket_name"):
            self.input_socket_name = self.custom_input_socket_name
        if hasattr(self, "custom_output_socket_name"):
            self.output_socket_name = self.custom_output_socket_name
            
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
        Get the data from the connected input socket specified by 'index'.

        :param index: The index of the input socket to get data from.
        :type index: int
        :return: The data from the connected input socket, or None if the socket is not connected.
        :rtype: Any or None
        """


        try:
            if len(self.getInputs(index)) > 0:
                node, new_index = self.getInput(index)
                #print("node", node)
                return node.getOutput(new_index, index)
            else:
                return None
        except Exception as e:
            done = handle_ainodes_exception()
            print(f"Error in getInputData: {e}")
            return None

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
        """if gs.should_run:
            if not self.busy:
                self.busy = True
                thread = WorkerThread(target=self.evalImplementationThreadHandler,
                                      on_finished=self.onWorkerFinished)
                thread.start()
            else:
                return
        else:
            gs.should_run = True
            return"""
        if gs.should_run:
            if not self.busy:
                self.busy = True
                worker = Worker(self.evalImplementationThreadHandler)
                worker.signals.result.connect(self.onWorkerFinished)
                self.scene.threadpool.start(worker)
                return
            else:
                return
        else:
            return

        """if self.busy == False:
            self.busy = True
            worker = Worker(self.evalImplementation_thread)
            worker.signals.result.connect(self.onWorkerFinished)
            #self.worker.setAutoDelete(True)
            self.scene.threadpool.start(worker)
            return
        else:
            return"""

    def evalImplementationThreadHandler(self, *args, **kwargs):
        try:
            result = self.evalImplementation_thread()
            return result
        except:
            handle_ainodes_exception()
            return None

    def evalImplementation_thread(self):
        return None

    def clearOutputs(self):
        ports = list(range(len(self.outputs) - 1))
        for port in ports:
            self.setOutput(port, None)

    def onWorkerFinished(self, result, exec=True):

        self.busy = False
        if result is not None:

            if hasattr(self, "output_data_ports"):
                ports = self.output_data_ports

            else:
                ports = list(range(len(self.outputs) - 1))

            x = 0
            for port in ports:
                if result[x] is not None:
                    self.markDirty(False)

                self.setOutput(port, result[x])
                x += 1

        # self.content.update()
        # self.content.finished.emit()
        if exec:
            if hasattr(self, "exec_port"):
                port = self.exec_port
            else:
                port = len(self.outputs) - 1
            if len(self.getOutputs(port)) > 0:
                self.executeChild(output_index=port)

    def eval(self, index=0):
        try:
            self.content.eval_signal.emit()
        except Exception as e:
            print(e, self)


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
            #self.markDescendantsDirty()
        except Exception as e:
            self.markInvalid()
            self.grNode.setToolTip(str(e))
            dumpException(e)


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
                #node.markDirty(True)
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
        #self.markDirty(True)

    def update_vars(self, data):


        if data is not None:
            try:
                for key, value in data.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
                return None
            except:
                handle_ainodes_exception()
                return None

    def serialize(self):
        """
        Serialize the node's data into a dictionary.

        Returns:
            dict: The serialized data of the node.
        """
        res = super().serialize()
        res['op_code'] = self.__class__.op_code
        res['content_label_objname'] = self.__class__.content_label_objname
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
        #print("Deserialized AiNode '%s'" % self.__class__.__name__, "res:", res)
        return res
    def remove(self):
        """
         Remove the node, clearing the values in its output sockets.
         """
        x = 0

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


        # if len(self.inputs) == 0:
        #     return True
        if hasattr(self, "force_run"):
            if self.force_run:
                return True
        for socket in self.inputs:
            if socket.is_input and socket.hasAnyEdge():
                for edge in socket.edges:
                    # if edge.end_socket.node.isDirty():
                    #     return False
                    #print(self, edge.start_socket.node, edge.end_socket.node)
                    if edge.start_socket.node is not self:
                        if edge.start_socket.node.isDirty():

                            return False
                    if edge.end_socket.node is not self:
                        if edge.end_socket.node.isDirty():
                            return False
        return True

        # for socket in self.inputs:
        #     for edge in socket.edges:
        #         if hasattr(edge, 'start_socket'):
        #             if hasattr(edge.start_socket, 'node'):
        #                 if edge.start_socket.node != self:
        #                     if edge.start_socket.node.isDirty():
        #                         #print(f"Node {self} cannot run because connected node {edge.start_socket.node} is dirty.")
        #                         return False
        #         elif hasattr(edge, 'end_socket'):
        #             if hasattr(edge.end_socket, 'node'):
        #                 if edge.end_socket.node != self:
        #                     if edge.end_socket.node.isDirty():
        #                         #print(f"Node {self} cannot run because connected node {edge.end_socket.node} is dirty.")
        #
        #                         return False
        # return True

class AiApiNode(AiNode):
    icon = ""
    op_code = 0
    op_title = "Undefined"
    content_label = ""
    content_label_objname = "calc_node_bg"
    category = "default"
    input_socket_name = ["EXEC"]
    output_socket_name = ["EXEC"]
    help_text = "Default help text"
    GraphicsNode_class = CalcGraphicsNode
    NodeContent_class = CalcContent

    def __init__(self, scene, inputs=[2,2], outputs=[1]):
        """
         Initialize the AiNode class with a scene, inputs, and outputs.

         Args:
             scene (QGraphicsScene): The scene where the node is displayed.
             inputs (list): A list of input sockets, with values representing their types.
                           1: EXEC
                           2: LATENT
                           3: CONDITIONING
                           5: IMAGE
                           6: DATA
                           (4 is not used yet)
                           Defaults to [2,2].
             outputs (list): A list of output sockets, with values representing their types.
                           1: EXEC
                           2: LATENT
                           3: CONDITIONING
                           5: IMAGE
                           6: DATA
                           (4 is not used yet)
                           Defaults to [1].
         """
        super().__init__(scene, self.__class__.op_title, inputs, outputs)
        self.set_socket_names()
        self.value = None
        self.output_values = {}
        # it's really important to mark all nodes Dirty by default
        self.markDirty()
        self.values = {}
        #self.task_queue = Queue()
        pass

    #@QtCore.Slot()
    def evalImplementation(self, index=0, *args, **kwargs):
        if self.busy == False:
            self.busy = True
            worker = Worker(self.evalImplementation_thread)
            worker.signals.result.connect(self.onWorkerFinished)
            self.scene.threadpool.start(worker)
        return None

    def evalImplementation_thread(self):
        print(f"PLEASE IMPLEMENT evalImplementation_thread function for {self}")
        pass
    #@QtCore.Slot(object)
    def onWorkerFinished(self):
        print(f"PLEASE IMPLEMENT onWorkerFinished function for {self}")
        pass
    def eval(self, index=0):
        try:
            #self.markDirty(True)
            self.content.eval_signal.emit()
        except Exception as e:
            print(e, self)


class AiDummyNode(Node):
    icon = ""
    op_code = 0
    op_title = "Undefined"
    content_label = ""
    content_label_objname = "calc_node_bg"
    category = "default"
    input_socket_name = ["EXEC"]
    output_socket_name = ["EXEC"]
    help_text = "Default help text"
    GraphicsNode_class = CalcGraphicsNode
    NodeContent_class = CalcContent
    sockets = None

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
                           5: IMAGE
                           6: DATA
                           (4 is not used yet)
                           Defaults to [2,2].
             outputs (list): A list of output sockets, with values representing their types.
                           1: EXEC
                           2: LATENT
                           3: CONDITIONING
                           5: IMAGE
                           6: DATA
                           (4 is not used yet)
                           Defaults to [1].
         """
        super().__init__(scene, self.__class__.op_title, inputs, outputs)
        self.set_socket_names()
        self.value = None
        self.output_values = {}
        self.values = {}
        self.busy = False
        self.init_done = None

    def set_socket_names(self):
        """
        Internal function to set socket names, override in your custom node pack to add additional socket types
        """

        return
    def update_all_sockets(self):
        return
        for socket in self.outputs + self.inputs:
            socket.setSocketPosition()
        self.updateConnectedEdges()

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
        return
        """
        Set the value of the output socket with the given index.

        Args:
            index (int): The index of the output socket.
            value: The value to be set for the output socket.
        """
        object_name = self.getID(index)

        self.values[object_name] = value

    def getOutput(self, index):
        return
        """
         Get the value of the output socket with the given index.

         Args:
             index (int): The index of the output socket.

         Returns:
             The value of the output socket, or None if it does not exist.
         """
        object_name = self.getID(index)
        try:
            return self.values[object_name]
        except:
            done = handle_ainodes_exception()

            print(f"Value doesnt exist yet, make sure to validate the node: {self.op_title}")
            return None

    def getInputData(self, index=0):

        """
        Get the data from the connected input socket specified by 'index'.

        :param index: The index of the input socket to get data from.
        :type index: int
        :return: The data from the connected input socket, or None if the socket is not connected.
        :rtype: Any or None
        """
        try:
            if len(self.getInputs(index)) > 0:
                node, index = self.getInput(index)
                return node.getOutput(index)
                return data
            else:
                return None
        except Exception as e:
            done = handle_ainodes_exception()

            print(f"Error in getInputData: {e}")
            return None

    def initSettings(self):
        """
        Initialize settings for the node, such as input and output socket positions.
        """
        super().initSettings()
        self.input_socket_position = LEFT_BOTTOM
        self.output_socket_position = RIGHT_BOTTOM

    #@QtCore.Slot()
    def evalImplementation(self, index=0, *args, **kwargs):
        return
    def evalImplementationThreadHandler(self, *args, **kwargs):
        return

    #@QtCore.Slot()
    def evalImplementation_thread(self):
        return None

    #@QtCore.Slot(object)
    def onWorkerFinished(self, result, exec=True):
        return

    def eval(self, index=0):
        return

    def executeChild(self, output_index=0):
        """
        Execute the child node connected to the output socket with the given index.

        Args:
            output_index (int): The index of the output socket. Defaults to 0.

        Returns:
            None
        """
        return

    def onInputChanged(self, socket=None):
        """
        Handle the event when an input socket value is changed.

        Args:
            socket: The input socket that changed.
        """
        print("%s::__onInputChanged" % self.__class__.__name__)
        self.markDirty(True)

    def update_vars(self, data):

        return

    def serialize(self):
        """
        Serialize the node's data into a dictionary.

        Returns:
            dict: The serialized data of the node.
        """
        res = super().serialize()
        res['op_code'] = self.__class__.op_code
        res['content_label_objname'] = self.__class__.content_label_objname
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

        self.values = None
        self.values = {}
        super().remove()



class AiApiNode(AiNode):
    icon = ""
    op_code = 0
    op_title = "Undefined"
    content_label = ""
    content_label_objname = "calc_node_bg"
    category = "default"
    input_socket_name = ["EXEC"]
    output_socket_name = ["EXEC"]
    help_text = "Default help text"
    GraphicsNode_class = CalcGraphicsNode
    NodeContent_class = CalcContent

    def __init__(self, scene, inputs=[2, 2], outputs=[1]):
        """
         Initialize the AiNode class with a scene, inputs, and outputs.

         Args:
             scene (QGraphicsScene): The scene where the node is displayed.
             inputs (list): A list of input sockets, with values representing their types.
                           1: EXEC
                           2: LATENT
                           3: CONDITIONING
                           5: IMAGE
                           6: DATA
                           (4 is not used yet)
                           Defaults to [2,2].
             outputs (list): A list of output sockets, with values representing their types.
                           1: EXEC
                           2: LATENT
                           3: CONDITIONING
                           5: IMAGE
                           6: DATA
                           (4 is not used yet)
                           Defaults to [1].
         """
        super().__init__(scene, self.__class__.op_title, inputs, outputs)
        self.set_socket_names()
        self.value = None
        self.output_values = {}
        # it's really important to mark all nodes Dirty by default
        self.markDirty()
        self.values = {}
        # self.task_queue = Queue()
        pass

    #@QtCore.Slot()
    def evalImplementation(self, index=0, *args, **kwargs):
        if self.busy == False:
            self.busy = True
            worker = Worker(self.evalImplementation_thread)
            worker.signals.result.connect(self.onWorkerFinished)
            self.scene.threadpool.start(worker)
        return None

    def evalImplementation_thread(self):
        print(f"PLEASE IMPLEMENT evalImplementation_thread function for {self}")
        pass

    #@QtCore.Slot(object)
    def onWorkerFinished(self):
        print(f"PLEASE IMPLEMENT onWorkerFinished function for {self}")
        pass

    def eval(self, index=0):
        try:
            # self.markDirty(True)
            self.content.eval_signal.emit()
        except Exception as e:
            print(e, self)



