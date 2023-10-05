import msgpack
from PyQt6.QtCore import QThreadPool, QRunnable, QUrl
from PyQt6.QtWebSockets import QWebSocket
from qtpy import QtCore

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

    #nodeResultReceived = QtCore.Signal(dict)


    def __init__(self, nodes, parent=None):
        self.nodes = nodes
        self.pool = QThreadPool()
        self.skipped_nodes = []
        self.processed_nodes = []  # New list to keep track of processed nodes
        self.process = True
        self.parent = parent
        self.running = False  # Add a flag to track if the runner is currently running
        #self.setup_api_connection()
        #self.nodeResultReceived.connect(self.process_node_message)

    def process_node_message(self, msg):
        print("API MESSAGE", msg)


    def setup_api_connection(self):
        self.websocket = QWebSocket()

        self.websocket.connected.connect(self.on_connected)
        self.websocket.disconnected.connect(self.on_disconnected)
        self.websocket.textMessageReceived.connect(self.on_text_message_received)
        self.websocket.binaryMessageReceived.connect(self.on_binary_message_received)

        self.websocket.open(QUrl('ws://localhost:8000/ws/process_nodes'))

    def on_connected(self):
        print("WebSocket connected!")

    def on_disconnected(self):
        print("WebSocket disconnected!")

    def on_text_message_received(self, message):
        # Handle text messages if needed
        pass

    def on_binary_message_received(self, message):
        data = msgpack.unpackb(message, raw=False)
        self.process_node_message(data)

    def collect_nodes_to_json(self):
        # This is a stub. You'll need to implement how you collect the node params.
        data = {}

        self.get_starting_nodes()

        for node in self.starting_nodes:
            data[node.id] = {}
            data[node.id]["params"] = node.content.serialize()
            data[node.id]["input_nodes"] = [node.id for node in node.inputs]
        # for node in self.nodes:
        #     data[node.name] = node.params  # Example, adjust accordingly
        return data
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
            # 2. Nodes that can run and are not ImagePreviewNode types
            # 3. Image preview nodes that can run and all their input nodes are not dirty
            # 4. Nodes that can run
            # 5. Image preview nodes
            # 6. Subgraph nodes
            return (
                not part_of_subgraph,
                is_image_preview and not can_run,
                not (is_image_preview and can_run and all_inputs_not_dirty),
                not can_run,
                not is_image_preview,
                is_subgraph
            )

        self.starting_nodes.sort(key=sorting_key)

    def run_next(self):
        if not self.exec:  # Check if execution should stop
            self.running = False
            return

        if not self.starting_nodes:
            # If there are no more nodes to process, check if we should loop or finish
            # Handle looping or finishing here...
            if self.loop:
                self.running = False
                self.start(loop=self.loop)
            return
        self.reorder_nodes()

        node_to_run = self.starting_nodes.pop(0)  # Process the first node in the list
        self.processed_nodes.append(node_to_run)  # Mark the node as processed

        print(f"Processing Node: {node_to_run}")
        worker = NodeWorker(node_to_run)

        def on_node_finished():
            node_to_run.markInvalid(False)
            node_to_run.markDirty(False)
            node_to_run.content.finished.disconnect(on_node_finished)

            # for output_socket in node_to_run.outputs:
            #     for edge in output_socket.edges:
            #         downstream_node = edge.end_socket.node
            #         if downstream_node not in self.starting_nodes and downstream_node not in self.skipped_nodes and downstream_node not in self.processed_nodes:
            #             if downstream_node.can_run():
            #                 self.starting_nodes.append(downstream_node)
            #             else:
            #                 self.skipped_nodes.append(downstream_node)

            self.parent.getView().update()
            self.run_next()  # Recursive call

        node_to_run.content.finished.connect(on_node_finished)

        node_to_run.markInvalid(True)
        node_to_run.content.update()

        self.pool.start(worker)
    def get_starting_nodes(self):

        # Mark nodes with 'seed' as dirty
        for node in self.parent.nodes:
            if hasattr(node.content, 'seed') and node.content.isVisible() == True:
                node.markDirty()
            if hasattr(node, 'make_dirty'):

                if node.make_dirty and node.content.isVisible() == True:
                    node.markDirty()
        # Prepare the initial list of starting nodes
        self.starting_nodes = [node for node in self.parent.nodes if
                               isinstance(node, AiNode) and node.isDirty() and node not in self.processed_nodes
                               and node.content.isVisible() == True]
        #print("starting nodes",self.starting_nodes)

    def start(self, loop=False):
        use_api = False
        if use_api:
            node_data = self.collect_nodes_to_json()

            packed_data = msgpack.packb(dict(node_data))

            self.websocket.sendBinaryMessage(packed_data)


            # packed_data = msgpack.packb(node_data, use_bin_type=False)
            # self.websocket.sendBinaryMessage(packed_data)
        else:

            self.stop()
            if self.running:  # If already running, simply return
                return
            self.exec = True
            self.loop = loop
            self.running = True
            gs.prefs.autorun = True
            # Clear the processed nodes
            self.processed_nodes.clear()

            self.get_starting_nodes()

            # Check for SubgraphNode instances and add their nodes to starting_nodes
            for node in self.parent.nodes:
                from ai_nodes.ainodes_engine_base_nodes.subgraph_nodes.subgraph_node import SubgraphNode
                if isinstance(node, SubgraphNode):  # Assuming 'nodes' is the attribute that holds the list of nodes for a SubgraphNode
                    nodes_to_check = node.get_nodes()
                    for node in nodes_to_check:
                        self.starting_nodes.append(node)
            # Reorder nodes to prioritize nodes that can run
            self.reorder_nodes()

            # Start processing
            self.run_next()

    def stop(self):
        self.running = False
        self.exec = False
