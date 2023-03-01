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
        self.add_input('in_exe')
        self.create_property('in_exe', None)
    def execute(self):
        #print("Loop node reached", self._graph)
        #self._graph.startsignal.emit()
        self._graph.start_signal_function()
        #self.runsignal.emit()
    @QtCore.Slot()
    def emit_run_signal(self):
        return
        self._graph.process_nodes_thread()
