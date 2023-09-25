from PyQt6.QtCore import QThreadPool, QRunnable

from ainodes_frontend.base import AiNode


class NodeWorker(QRunnable):
    def __init__(self, node):
        super().__init__()
        self.node = node

    def run(self):

        # print(self.node)
        # self.node.markDirty(False)

        self.node.onWorkerFinished(result=self.node.evalImplementation_thread(), exec=False)
        self.node.content.update()
        self.node.content.finished.emit()


class NodeRunner:

    def __init__(self, nodes, parent=None):
        self.nodes = nodes
        self.pool = QThreadPool()
        self.skipped_nodes = []
        self.processed_nodes = []  # New list to keep track of processed nodes
        self.process = True
        self.parent = parent

    def reorder_nodes(self):
        """Reorder starting_nodes to prioritize nodes that can run."""
        self.starting_nodes.sort(key=lambda node: not node.can_run())

    def run_next(self):
        if not self.starting_nodes:
            self.starting_nodes = [node for node in self.skipped_nodes if node.can_run() and node.isDirty() and node not in self.processed_nodes]
            self.skipped_nodes = [node for node in self.skipped_nodes if not (node.can_run() and node.isDirty()) and node not in self.processed_nodes]

            # If still no nodes can be processed, return
            if not self.starting_nodes:
                return

        # Reorder nodes to prioritize nodes that can run
        self.reorder_nodes()

        node_to_run = self.starting_nodes.pop(0)  # Since we've reordered, the first node can run
        self.processed_nodes.append(node_to_run)  # Mark the node as processed

        print(f"Processing Node: {node_to_run}")
        worker = NodeWorker(node_to_run)

        def on_node_finished():
            # print(f"Finished processing node: {node_to_run}")
            node_to_run.markDirty(False)
            node_to_run.content.finished.disconnect(on_node_finished)

            for output_socket in node_to_run.outputs:
                for edge in output_socket.edges:
                    downstream_node = edge.end_socket.node
                    if downstream_node not in self.starting_nodes and downstream_node not in self.skipped_nodes and downstream_node not in self.processed_nodes:
                        if downstream_node.can_run():
                            # print(f"Downstream node {downstream_node} can run. Adding to starting nodes.")
                            self.starting_nodes.append(downstream_node)
                        else:
                            # print(f"Downstream node {downstream_node} cannot run yet. Skipping for now.")
                            self.skipped_nodes.append(downstream_node)
            self.parent.getView().update()
            self.run_next()  # Recursive call

        node_to_run.content.finished.connect(on_node_finished)
        self.pool.start(worker)

    def start(self):
        # print("Trying to start")
        self.processed_nodes.clear()

        for node in self.parent.nodes:
            if hasattr(node.content, 'seed'):
                node.markDirty()

        self.starting_nodes = [node for node in self.parent.nodes if
                               isinstance(node, AiNode) and node.isDirty() and node not in self.processed_nodes]
        # print(self.starting_nodes)

        self.run_next()
