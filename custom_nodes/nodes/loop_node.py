from NodeGraphQt import BaseNode
from Qt import QtCore

from custom_nodes.auto_base_node import AutoBaseNode


class LoopNode(AutoBaseNode):
    """
    An example of a node with a embedded QLineEdit.
    """

    # unique node identifier.
    __identifier__ = 'nodes.widget'

    # initial default node name.
    NODE_NAME = 'loop'
    def __init__(self, parent=None):
        super(LoopNode, self).__init__()

        # create input & output ports
        self.add_input('in')
        self.create_property('in', None)
    def execute(self):
        #print("Loop node reached", self._graph)
        self._graph.process_nodes_thread()
        #self.emit_run_signal()
        #self.runsignal.emit()
    @QtCore.Slot()
    def emit_run_signal(self):
        self._graph.process_nodes_thread()
