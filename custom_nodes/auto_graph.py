from NodeGraphQt.base.commands import NodeRemovedCmd
from Qt import QtCore
import os, json

from NodeGraphQt import NodeGraph, GroupNode

from custom_nodes.auto_base_node import BaseNode
from worker.worker import Worker


class AutoGraph(NodeGraph):
    """
    The ``NodeGraphQt.BaseNode`` class is the base class for nodes that allows
    port connections from one node to another.

    **Inherited from:** :class:`NodeGraphQt.NodeObject`

    .. image:: ../_images/node.png
        :width: 250px

    example snippet:

    .. code-block:: python
        :linenos:

        from NodeGraphQt import BaseNode

        class ExampleNode(BaseNode):

            # unique node identifier domain.
            __identifier__ = 'io.jchanvfx.github'

            # initial default node name.
            NODE_NAME = 'My Node'

            def __init__(self):
                super(ExampleNode, self).__init__()

                # create an input port.
                self.add_input('in')

                # create an output port.
                self.add_output('out')
    """

    NODE_NAME = 'Node'
    startsignal = QtCore.Signal()
    loopsignal = QtCore.Signal()
    def __init__(self):
        super(AutoGraph, self).__init__()
        self.startsignal.connect(self.process_nodes)
        self.loopsignal.connect(self.process_nodes_thread)
        self.threadpool = QtCore.QThreadPool()
    @QtCore.Slot()
    def process_nodes(self):
        self.worker = Worker(self.process_nodes_thread)
        self.threadpool.start(self.worker)

    def process_nodes_thread(self, progress_callback=None):
        self.all_nodes()[0].execute()

    def start_signal_function(self):
        self.startsignal.emit()
    def loop_signal_function(self):
        self.loopsignal.emit()

    def delete_node(self, node, push_undo=True):
        """
        Remove the node from the node graph.

        Args:
            node (NodeGraphQt.BaseNode): node object.
            push_undo (bool): register the command to the undo stack. (default: True)
        """
        #print(node.name())
        #assert isinstance(node, NodeObject), \
        #    'node must be a instance of a NodeObject.'
        node_id = node.id
        if push_undo:
            self._undo_stack.beginMacro('delete node: "{}"'.format(node.name()))

        if isinstance(node, BaseNode):
            for p in node.input_ports():
                if p.locked():
                    p.set_locked(False,
                                 connected_ports=False,
                                 push_undo=push_undo)
                p.clear_connections(push_undo=push_undo)
            for p in node.output_ports():
                if p.locked():
                    p.set_locked(False,
                                 connected_ports=False,
                                 push_undo=push_undo)
                p.clear_connections(push_undo=push_undo)

        # collapse group node before removing.
        if isinstance(node, GroupNode) and node.is_expanded:
            node.collapse()

        if push_undo:
            self._undo_stack.push(NodeRemovedCmd(self, node))
            self._undo_stack.endMacro()
        else:
            NodeRemovedCmd(self, node).redo()

        self.nodes_deleted.emit([node_id])

    def run_node_by_id(self, node_id):
        node = self._model.nodes.get(node_id, None)
        worker = Worker(node.execute)
        self.threadpool.start(worker)