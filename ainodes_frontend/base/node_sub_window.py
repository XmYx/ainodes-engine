import os

from PySide6 import QtWidgets
from qtpy.QtGui import QContextMenuEvent, QBrush
from qtpy.QtWidgets import QColorDialog
from qtpy import QtCore

from ainodes_frontend.base.tab_search import TabSearchMenuWidget
from ainodes_frontend.node_engine.node_graphics_node import QDMGraphicsBGNode
from qtpy.QtGui import QIcon, QPixmap
from qtpy.QtCore import QDataStream, QIODevice, Qt, QThreadPool
from qtpy.QtWidgets import QAction, QGraphicsProxyWidget, QMenu, QOpenGLWidget

from ainodes_frontend.node_engine.node_node import Node
from ainodes_frontend.base.node_config import CALC_NODES, get_class_from_opcode, LISTBOX_MIMETYPE, \
    node_categories, get_class_from_content_label_objname
from ainodes_frontend.node_engine.node_editor_widget import NodeEditorWidget
from ainodes_frontend.node_engine.node_edge import EDGE_TYPE_DIRECT, EDGE_TYPE_BEZIER, EDGE_TYPE_SQUARE
from ainodes_frontend.node_engine.node_graphics_view import MODE_EDGE_DRAG
from ainodes_frontend.node_engine.utils import dumpException, loadStylesheets

DEBUG = False
DEBUG_CONTEXT = False


class CalculatorSubWindow(NodeEditorWidget):
    def __init__(self):
        super().__init__()
        self.context_menu_style = 'modern'

        self.setAttribute(Qt.WA_DeleteOnClose)

        self.setTitle()

        self.initNewNodeActions()
        #self.scene.threadpool = QThreadPool()
        self.scene.addHasBeenModifiedListener(self.setTitle)
        self.scene.history.addHistoryRestoredListener(self.onHistoryRestored)
        self.scene.addDragEnterListener(self.onDragEnter)
        self.scene.addDropListener(self.onDrop)
        self.scene.setNodeClassSelector(self.getNodeClassFromData)

        self._close_event_listeners = []

        self.run_all_action = QAction("Run All")
        self.run_all_action.triggered.connect(self.doRunAll)
        """self.stylesheet_filename = os.path.join(os.path.dirname(__file__), "ainodes_frontend/qss/nodeeditor-dark.qss")
        loadStylesheets(
            os.path.join(os.path.dirname(__file__), "ainodes_frontend/qss/nodeeditor-dark.qss"),
            self.stylesheet_filename
        )"""
        #self._search_widget = TabSearchMenuWidget()
        #self._search_widget.search_submitted.connect(self._on_search_submitted)
        #print(self._search_widget.isVisible())
        self.scenePos = None
        #self.tab_search_toggle()

    def tab_search_toggle(self):
        state = self._search_widget.isVisible()
        if state == False:
            self._search_widget.setVisible(state)
            print(self._search_widget.isVisible())
            #self.setFocus()
            #return
        state = True
        if self.scenePos:
            print(self.scenePos.y())

            pos = QtCore.QPoint(int(self.width() / 2),
                                               int(self.height() / 2))
            print(pos)
            rect = self._search_widget.rect()
            #new_pos = QtCore.QPoint(int(pos.x() - rect.width() / 2),
            #                        int(pos.y() - rect.height() / 2))
            print(pos)
            self._search_widget.move(pos)
            self._search_widget.setVisible(state)
            self._search_widget.setFocus()

            rect = self.scene.getView().mapToScene(rect).boundingRect().toRect()
            self.scene.getView().update(rect)


    def handle_task_finished(self):
        self.scene.queue.start_next_task()
    def getNodeClassFromData(self, data):
        if 'op_code' not in data: return Node

        #print("data", data['content_label_objname'])
        #print("op_code", get_class_from_opcode(data['op_code']))
        return get_class_from_content_label_objname(data['content_label_objname'])
        #return get_class_from_opcode(data['op_code'])

    def doEvalOutputs(self):
        # eval all output nodes
        for node in self.scene.nodes:
            if node.__class__.__name__ == "CalcNode_Output":
                node.eval()
    def doRunAll(self):
        # eval all output nodes
        for node in self.scene.nodes:
            node.markDirty(True)
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
        keys = list(CALC_NODES.keys())
        keys.sort()
        for key in keys:
            node = CALC_NODES[key]
            self.node_actions[node.op_code] = QAction(QIcon(node.icon), node.op_title)
            self.node_actions[node.op_code].setData(node.op_code)

    def initNodesContextMenu_orig(self):
        context_menu = QMenu(self)
        keys = list(CALC_NODES.keys())
        keys.sort()
        for key in keys: context_menu.addAction(self.node_actions[key])
        context_menu.addAction(self.run_all_action)
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
            op_code = dataStream.readInt8()
            text = dataStream.readQString()

            mouse_position = event.pos()
            scene_position = self.scene.grScene.views()[0].mapToScene(mouse_position)

            if DEBUG: print("GOT DROP: [%d] '%s'" % (op_code, text), "mouse:", mouse_position, "scene:", scene_position)

            try:
                node = get_class_from_opcode(op_code)(self.scene)
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

        item = self.scene.getItemAt(event.pos())

        #print("ITEM:", item)
        if isinstance(item, QDMGraphicsBGNode):
            context_menu = QMenu()
            set_color_action = context_menu.addAction("Set Color")

            action = context_menu.exec_(self.mapToGlobal(event.pos()))

            if action == set_color_action:
                color_dialog = QColorDialog()
                color = color_dialog.getColor()

                if color.isValid():
                    item._brush_background = QBrush(color)
                    item.update()
                    return


        context_menu = QMenu(self)
        markDirtyAct = context_menu.addAction("Mark Dirty")
        markDirtyDescendantsAct = context_menu.addAction("Mark Descendant Dirty")
        markInvalidAct = context_menu.addAction("Mark Invalid")
        unmarkInvalidAct = context_menu.addAction("Unmark Invalid")
        evalAct = context_menu.addAction("Eval")
        helpAct = context_menu.addAction("Help")



        action = context_menu.exec_(self.mapToGlobal(event.pos()))

        selected = None

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
        if selected and action == helpAct:
            selected.showNiceDialog()


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

    def initNodesContextMenu(self, event):
        menu = QMenu()

        # create submenus for categories
        category_menus = {}
        for category in node_categories:
            submenu = QMenu(category.capitalize(), menu)
            category_menus[category] = submenu

        # sort the category menus by keys
        sorted_category_menus = dict(sorted(category_menus.items()))

        # add sorted category menus to the root menu
        for submenu in sorted_category_menus.values():
            menu.addMenu(submenu)

        # add nodes to corresponding submenu
        keys = list(CALC_NODES.keys())
        keys.sort()
        for key in keys:
            node = get_class_from_opcode(key)
            action = QAction(node.op_title, self)
            action.setData(node.op_code)
            pixmap = QPixmap(node.icon if node.icon is not None else ".")
            action.setIcon(QIcon(pixmap))

            # add action to the corresponding submenu
            category_menus[node.category].addAction(action)

        # sort actions in each category submenu alphabetically
        for submenu in category_menus.values():
            submenu.addActions(sorted(submenu.actions(), key=lambda action: action.text()))

        # add "Run All" action to the root menu
        #run_all_action = QAction("Run All", self)
        #menu.addAction(run_all_action)

        return menu

    def init_nodes_list_widget(self, event):
        nodes_dialog = NodeListDialog()
        nodes_dialog.setFixedHeight(1024)

        x = self.mapToGlobal(event.pos()).x() - 256
        y = self.mapToGlobal(event.pos()).y() - 512

        nodes_dialog.setGeometry(x, y, 512, 1024)

        # create a dictionary to store category items
        category_items = {}
        for category in node_categories:
            category_items[category] = []

        # sort nodes and add them to the corresponding category
        keys = list(CALC_NODES.keys())
        keys.sort()
        for key in keys:
            node = get_class_from_opcode(key)
            pixmap = QPixmap(node.icon if node.icon is not None else ".")
            category_items[node.category].append((node.op_title, node.op_code, QIcon(pixmap)))

        # sort category items alphabetically and add them to the list widget
        for items in category_items.values():
            items.sort(key=lambda item: item[0])
            for item in items:
                nodes_dialog.add_item(item[0], item[1], item[2])

        return nodes_dialog
    def handleNewNodeContextMenu_orig(self, event):

        if DEBUG_CONTEXT: print("CONTEXT: EMPTY SPACE")
        context_menu = self.initNodesContextMenu(event)
        action = context_menu.exec_(self.mapToGlobal(event.pos()))

        if action is not None:

            if action.text() != "Run All":
                new_calc_node = get_class_from_opcode(action.data())(self.scene)
                scene_pos = self.scene.getView().mapToScene(event.pos())
                new_calc_node.setPos(scene_pos.x(), scene_pos.y())
                if DEBUG_CONTEXT: print("Selected node:", new_calc_node)

                if self.scene.getView().mode == MODE_EDGE_DRAG:
                    # if we were dragging an edge...
                    target_socket = self.determine_target_socket_of_node(self.scene.getView().dragging.drag_start_socket.is_output, new_calc_node)
                    if target_socket is not None:
                        self.scene.getView().dragging.edgeDragEnd(target_socket.grSocket)
                        self.finish_new_node_state(new_calc_node)

                else:
                    self.scene.history.storeHistory("Created %s" % new_calc_node.__class__.__name__)

    def handleNewNodeContextMenu(self, event):
        if self.context_menu_style == 'modern':
            self.handleNewNodeContextMenuFunction(event)
        else:
            self.handleNewNodeContextMenu_orig(event)
    def handleNewNodeContextMenuFunction(self, event):
        if DEBUG_CONTEXT: print("CONTEXT: EMPTY SPACE")
        nodes_dialog = self.init_nodes_list_widget(event)
        result = nodes_dialog.exec_()

        if result == QtWidgets.QDialog.Accepted:
            selected_op_code = nodes_dialog.selected_data()
            if selected_op_code is not None:
                new_calc_node = get_class_from_opcode(selected_op_code)(self.scene)
                scene_pos = self.scene.getView().mapToScene(event.pos())
                new_calc_node.setPos(scene_pos.x(), scene_pos.y())
                if DEBUG_CONTEXT: print("Selected node:", new_calc_node)

                if self.scene.getView().mode == MODE_EDGE_DRAG:
                    # if we were dragging an edge...
                    target_socket = self.determine_target_socket_of_node(
                        self.scene.getView().dragging.drag_start_socket.is_output, new_calc_node)
                    if target_socket is not None:
                        self.scene.getView().dragging.edgeDragEnd(target_socket.grSocket)
                        self.finish_new_node_state(new_calc_node)

                else:
                    self.scene.history.storeHistory("Created %s" % new_calc_node.__class__.__name__)
    def mouseMoveEvent(self, event):
        super(CalculatorSubWindow, self).mouseMoveEvent(event)
        #self.pos = QtCore.QPointF(event.screenPos())
        self.scenePos = event.scenePos()
        print(self.scenePos)
class NodeListDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(NodeListDialog, self).__init__(parent)
        self.setWindowTitle("Select a Node")
        self.layout = QtWidgets.QVBoxLayout()
        self.list_widget = QtWidgets.QListWidget()
        self.ok_button = QtWidgets.QPushButton("OK")
        self.ok_button.setEnabled(False)

        self.layout.addWidget(self.list_widget)
        self.layout.addWidget(self.ok_button)
        self.setLayout(self.layout)

        self.list_widget.itemDoubleClicked.connect(self.accept)
        self.list_widget.itemSelectionChanged.connect(self.enable_ok_button)
        self.ok_button.clicked.connect(self.accept)

    def enable_ok_button(self):
        self.ok_button.setEnabled(True)

    def add_item(self, text, data, icon):
        item = QtWidgets.QListWidgetItem(icon, text)
        item.setData(Qt.UserRole, data)
        self.list_widget.addItem(item)

    def selected_data(self):
        selected_items = self.list_widget.selectedItems()
        if selected_items:
            return selected_items[0].data(Qt.UserRole)
        return None