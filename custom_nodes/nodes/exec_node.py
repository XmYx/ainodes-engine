from NodeGraphQt import BaseNode
from Qt import QtCore

from custom_nodes.auto_base_node import AutoBaseNode


class ExecNode(AutoBaseNode):
    """
    An example of a node with a embedded QLineEdit.
    """

    # unique node identifier.
    __identifier__ = 'nodes.widget'

    # initial default node name.
    NODE_NAME = 'exec'
    def __init__(self, parent=None):
        super(ExecNode, self).__init__()

        # create input & output ports
        self.add_output('exec')
        self.create_property('exec', None)
        self.add_input('in_exe')
        self.create_property('in_exe', None)
    def execute(self):
        self._graph.start_signal_function()
