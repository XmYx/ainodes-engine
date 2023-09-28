from PyQt6.QtCore import QThreadPool, QRunnable

from ainodes_frontend.base import AiNode
from ainodes_frontend import singleton as gs

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

        # Check if the node is part of any SubgraphNode's nodes list
        from ai_nodes.ainodes_engine_base_nodes.subgraph_nodes.subgraph_node import SubgraphNode
        def is_part_of_subgraph(node):
            for n in self.parent.nodes:

                if isinstance(n, SubgraphNode) and hasattr(n, 'nodes') and node in n.nodes:
                    return True
            return False

        def sorting_key(node):
            from ai_nodes.ainodes_engine_base_nodes.image_nodes.image_preview_node import ImagePreviewNode

            is_image_preview = isinstance(node, ImagePreviewNode)
            is_subgraph = isinstance(node, SubgraphNode)
            part_of_subgraph = is_part_of_subgraph(node)
            can_run = node.can_run()

            # Check if all input nodes of the ImagePreviewNode are not dirty
            all_inputs_not_dirty = True
            if is_image_preview:
                for input_socket in node.inputs:
                    for edge in input_socket.edges:
                        if edge.start_socket.node.isDirty():
                            all_inputs_not_dirty = False
                            break

            # Prioritize by:
            # 1. Nodes that are part of a SubgraphNode's nodes list
            # 2. Image preview nodes that can run and all their input nodes are not dirty
            # 3. Nodes that can run
            # 4. Image preview nodes
            # 5. Subgraph nodes
            return (
                not part_of_subgraph,
                not (is_image_preview and can_run and all_inputs_not_dirty),
                not can_run,
                not is_image_preview,
                is_subgraph
            )

        self.starting_nodes.sort(key=sorting_key)

    def run_next(self):
        if not self.starting_nodes:
            self.starting_nodes = [node for node in self.skipped_nodes if node.can_run() and node.isDirty() and node not in self.processed_nodes]
            self.skipped_nodes = [node for node in self.skipped_nodes if not (node.can_run() and node.isDirty()) and node not in self.processed_nodes]

            # If still no nodes can be processed, return
            if not self.starting_nodes:
                gs.prefs.autorun = False
                return

        # Reorder nodes to prioritize nodes that can run
        self.reorder_nodes()

        node_to_run = self.starting_nodes.pop(0)  # Since we've reordered, the first node can run
        self.processed_nodes.append(node_to_run)  # Mark the node as processed

        #print(f"Processing Node: {node_to_run}")
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
        gs.prefs.autorun = True
        # Clear the processed nodes
        self.processed_nodes.clear()

        # Mark nodes with 'seed' as dirty
        for node in self.parent.nodes:
            if hasattr(node.content, 'seed'):
                node.markDirty()
            if hasattr(node, 'make_dirty'):
                if node.make_dirty:
                    node.markDirty()

        # Prepare the initial list of starting nodes
        self.starting_nodes = [node for node in self.parent.nodes if
                               isinstance(node, AiNode) and node.isDirty() and node not in self.processed_nodes]

        # Check for SubgraphNode instances and add their nodes to starting_nodes
        for node in self.parent.nodes:
            from ai_nodes.ainodes_engine_base_nodes.subgraph_nodes.subgraph_node import SubgraphNode
            if isinstance(node, SubgraphNode):  # Assuming 'nodes' is the attribute that holds the list of nodes for a SubgraphNode
                nodes_to_check = node.get_nodes()
                for node in nodes_to_check:
                    self.starting_nodes.append(node)

        # Start processing
        self.run_next()
