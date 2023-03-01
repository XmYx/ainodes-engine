from Qt import QtCore
import os, json

from NodeGraphQt import NodeGraph

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

