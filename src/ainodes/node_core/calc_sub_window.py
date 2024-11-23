from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import QDataStream, QIODevice, Qt
from PyQt5.QtWidgets import QAction, QGraphicsProxyWidget, QMenu

from node_core.node_register import NODE_CLASSES, get_class_from_name, LISTBOX_MIMETYPE
from nodeeditor.node_editor_widget import NodeEditorWidget
from nodeeditor.node_edge import EDGE_TYPE_DIRECT, EDGE_TYPE_BEZIER, EDGE_TYPE_SQUARE
from nodeeditor.node_graphics_view import MODE_EDGE_DRAG
from nodeeditor.utils import dumpException

DEBUG = False
DEBUG_CONTEXT = False


class CalculatorSubWindow(NodeEditorWidget):
    def __init__(self):
        super().__init__()
        # self.setAttribute(Qt.WA_DeleteOnClose)

        self.setTitle()

        self.initNewNodeActions()

        self.scene.addHasBeenModifiedListener(self.setTitle)
        self.scene.history.addHistoryRestoredListener(self.onHistoryRestored)
        self.scene.addDragEnterListener(self.onDragEnter)
        self.scene.addDropListener(self.onDrop)
        self.scene.setNodeClassSelector(self.getNodeClassFromData)

        self._close_event_listeners = []

    def getNodeClassFromData(self, data):
        #if 'op_code' not in data: return Node
        if data.get("content_label_objname"):
            return get_class_from_name(data['content_label_objname'])
        else:
            return Node

    def doEvalOutputs(self):
        # eval all output nodes
        for node in self.scene.nodes:
            if node.__class__.__name__ == "CalcNode_Output":
                node.eval()

    def onHistoryRestored(self):
        self.doEvalOutputs()

    def fileLoad(self, filename):
        if super().fileLoad(filename):
            self.doEvalOutputs()
            return True

        return False

    def initNewNodeActions(self):
        self.node_actions = {}
        keys = list(NODE_CLASSES.keys())
        keys.sort()
        for key in keys:
            node = NODE_CLASSES[key]
            self.node_actions[node.__class__.__name__] = QAction(QIcon(node.icon), node.op_title)
            self.node_actions[node.__class__.__name__].setData(node.__class__.__name__)

    def initNodesContextMenu(self):
        context_menu = QMenu(self)
        class_names = list(NODE_CLASSES.keys())
        class_names.sort()
        self.node_actions = {}  # Ensure node_actions is defined and clear previous data

        for class_name in class_names:
            action = QAction(class_name, self)
            action.setData(class_name)
            self.node_actions[class_name] = action
            context_menu.addAction(action)

        return context_menu

    def setTitle(self):
        self.setWindowTitle(self.getUserFriendlyFilename())

    def addCloseEventListener(self, callback):
        self._close_event_listeners.append(callback)

    def closeEvent(self, event):
        for callback in self._close_event_listeners: callback(self, event)

    def onDragEnter(self, event):
        if event.mimeData().hasFormat(LISTBOX_MIMETYPE):
            event.acceptProposedAction()
        else:
            # print(" ... denied drag enter event")
            event.setAccepted(False)

    def onDrop(self, event):
        if event.mimeData().hasFormat(LISTBOX_MIMETYPE):
            eventData = event.mimeData().data(LISTBOX_MIMETYPE)
            dataStream = QDataStream(eventData, QIODevice.ReadOnly)
            pixmap = QPixmap()
            dataStream >> pixmap
            class_name = dataStream.readQString()  # Read the class name instead of op_code
            text = dataStream.readQString()

            mouse_position = event.pos()
            scene_position = self.scene.grScene.views()[0].mapToScene(mouse_position)

            if DEBUG: print("GOT DROP: [%d] '%s'" % (class_name, text), "mouse:", mouse_position, "scene:", scene_position)

            try:
                node = get_class_from_name(class_name)(self.scene)
                node.setPos(scene_position.x(), scene_position.y())
                self.scene.history.storeHistory("Created node %s" % node.__class__.__name__)
            except Exception as e: dumpException(e)


            event.setDropAction(Qt.MoveAction)
            event.accept()
        else:
            # print(" ... drop ignored, not requested format '%s'" % LISTBOX_MIMETYPE)
            event.ignore()


    def contextMenuEvent(self, event):
        try:
            item = self.scene.getItemAt(event.pos())
            if DEBUG_CONTEXT: print(item)

            if type(item) == QGraphicsProxyWidget:
                item = item.widget()

            if hasattr(item, 'node') or hasattr(item, 'socket'):
                self.handleNodeContextMenu(event)
            elif hasattr(item, 'edge'):
                self.handleEdgeContextMenu(event)
            #elif item is None:
            else:
                self.handleNewNodeContextMenu(event)

            return super().contextMenuEvent(event)
        except Exception as e: dumpException(e)

    def handleNodeContextMenu(self, event):
        if DEBUG_CONTEXT: print("CONTEXT: NODE")
        context_menu = QMenu(self)
        markDirtyAct = context_menu.addAction("Mark Dirty")
        markDirtyDescendantsAct = context_menu.addAction("Mark Descendant Dirty")
        markInvalidAct = context_menu.addAction("Mark Invalid")
        unmarkInvalidAct = context_menu.addAction("Unmark Invalid")
        evalAct = context_menu.addAction("Eval")
        action = context_menu.exec_(self.mapToGlobal(event.pos()))

        selected = None
        item = self.scene.getItemAt(event.pos())
        if type(item) == QGraphicsProxyWidget:
            item = item.widget()

        if hasattr(item, 'node'):
            selected = item.node
        if hasattr(item, 'socket'):
            selected = item.socket.node

        if DEBUG_CONTEXT: print("got item:", selected)
        if selected and action == markDirtyAct: selected.markDirty()
        if selected and action == markDirtyDescendantsAct: selected.markDescendantsDirty()
        if selected and action == markInvalidAct: selected.markInvalid()
        if selected and action == unmarkInvalidAct: selected.markInvalid(False)
        if selected and action == evalAct:
            val = selected.eval()
            if DEBUG_CONTEXT: print("EVALUATED:", val)


    def handleEdgeContextMenu(self, event):
        if DEBUG_CONTEXT: print("CONTEXT: EDGE")
        context_menu = QMenu(self)
        bezierAct = context_menu.addAction("Bezier Edge")
        directAct = context_menu.addAction("Direct Edge")
        squareAct = context_menu.addAction("Square Edge")
        action = context_menu.exec_(self.mapToGlobal(event.pos()))

        selected = None
        item = self.scene.getItemAt(event.pos())
        if hasattr(item, 'edge'):
            selected = item.edge

        if selected and action == bezierAct: selected.edge_type = EDGE_TYPE_BEZIER
        if selected and action == directAct: selected.edge_type = EDGE_TYPE_DIRECT
        if selected and action == squareAct: selected.edge_type = EDGE_TYPE_SQUARE

    # helper functions
    def determine_target_socket_of_node(self, was_dragged_flag, new_calc_node):
        target_socket = None
        if was_dragged_flag:
            if len(new_calc_node.inputs) > 0: target_socket = new_calc_node.inputs[0]
        else:
            if len(new_calc_node.outputs) > 0: target_socket = new_calc_node.outputs[0]
        return target_socket

    def finish_new_node_state(self, new_calc_node):
        self.scene.doDeselectItems()
        new_calc_node.grNode.doSelect(True)
        new_calc_node.grNode.onSelected()

    def handleNewNodeContextMenu(self, event):
        if DEBUG_CONTEXT: print("CONTEXT: EMPTY SPACE")
        context_menu = self.initNodesContextMenu()
        action = context_menu.exec_(self.mapToGlobal(event.pos()))

        if action is not None:
            class_name = action.data()
            new_calc_node_class = get_class_from_name(class_name)
            new_calc_node = new_calc_node_class(self.scene)
            scene_pos = self.scene.getView().mapToScene(event.pos())
            new_calc_node.setPos(scene_pos.x(), scene_pos.y())
            if DEBUG_CONTEXT: print("Selected node:", new_calc_node)

            if self.scene.getView().mode == MODE_EDGE_DRAG:
                # If we were dragging an edge...
                target_socket = self.determine_target_socket_of_node(
                    self.scene.getView().dragging.drag_start_socket.is_output, new_calc_node)
                if target_socket is not None:
                    self.scene.getView().dragging.edgeDragEnd(target_socket.grSocket)
                    self.finish_new_node_state(new_calc_node)

            else:
                self.scene.history.storeHistory(f"Created {new_calc_node.__class__.__name__}")
