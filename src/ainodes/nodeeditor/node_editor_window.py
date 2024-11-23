# -*- coding: utf-8 -*-
"""
A module containing the Main Window class
"""
import os, json
from qtpy.QtCore import QSize, QSettings, QPoint
from qtpy.QtWidgets import QMainWindow, QLabel, QAction, QMessageBox, QFileDialog, QApplication
from nodeeditor.node_editor_widget import NodeEditorWidget


class NodeEditorWindow(QMainWindow):
    NodeEditorWidget_class = NodeEditorWidget

    """Class representing NodeEditor's Main Window"""
    def __init__(self):
        """
        :Instance Attributes:

        - **name_company** - name of the company, used for permanent profile settings
        - **name_product** - name of this App, used for permanent profile settings
        """
        super().__init__()

        self.name_company = 'Blenderfreak'
        self.name_product = 'NodeEditor'

        self.initUI()


    def initUI(self):
        """Set up this ``QMainWindow``. Create :class:`~nodeeditor.node_editor_widget.NodeEditorWidget`, Actions and Menus"""
        self.createActions()
        self.createMenus()

        # create node editor widget
        self.nodeeditor = self.__class__.NodeEditorWidget_class(self)
        self.nodeeditor.scene.addHasBeenModifiedListener(self.setTitle)
        self.setCentralWidget(self.nodeeditor)

        self.createStatusBar()

        # set window properties
        # self.setGeometry(200, 200, 800, 600)
        self.setTitle()
        self.show()

    def sizeHint(self):
        return QSize(800, 600)

    def createStatusBar(self):
        """Create Status bar and connect to `Graphics View` scenePosChanged event"""
        self.statusBar().showMessage("")
        self.status_mouse_pos = QLabel("")
        self.statusBar().addPermanentWidget(self.status_mouse_pos)
        self.nodeeditor.view.scenePosChanged.connect(self.onScenePosChanged)

    def createActions(self):
        """Create basic `File` and `Edit` actions"""
        self.actNew = QAction('&New', self, shortcut='Ctrl+N', statusTip="Create new graph", triggered=self.onFileNew)
        self.actOpen = QAction('&Open', self, shortcut='Ctrl+O', statusTip="Open file", triggered=self.onFileOpen)
        self.actSave = QAction('&Save', self, shortcut='Ctrl+S', statusTip="Save file", triggered=self.onFileSave)
        self.actSaveAs = QAction('Save &As...', self, shortcut='Ctrl+Shift+S', statusTip="Save file as...", triggered=self.onFileSaveAs)
        self.actExit = QAction('E&xit', self, shortcut='Ctrl+Q', statusTip="Exit application", triggered=self.close)

        self.actUndo = QAction('&Undo', self, shortcut='Ctrl+Z', statusTip="Undo last operation", triggered=self.onEditUndo)
        self.actRedo = QAction('&Redo', self, shortcut='Ctrl+Shift+Z', statusTip="Redo last operation", triggered=self.onEditRedo)
        self.actCut = QAction('Cu&t', self, shortcut='Ctrl+X', statusTip="Cut to clipboard", triggered=self.onEditCut)
        self.actCopy = QAction('&Copy', self, shortcut='Ctrl+C', statusTip="Copy to clipboard", triggered=self.onEditCopy)
        self.actPaste = QAction('&Paste', self, shortcut='Ctrl+V', statusTip="Paste from clipboard", triggered=self.onEditPaste)
        self.actDelete = QAction('&Delete', self, shortcut='Del', statusTip="Delete selected items", triggered=self.onEditDelete)
        self.actOpenComfyWorkflow = QAction('Open Comfy Workflow', self, statusTip='Open Comfy Workflow', triggered=self.onOpenComfyWorkflow)


    def createMenus(self):
        """Create Menus for `File` and `Edit`"""
        self.createFileMenu()
        self.createEditMenu()

    def createFileMenu(self):
        menubar = self.menuBar()
        self.fileMenu = menubar.addMenu('&File')
        self.fileMenu.addAction(self.actNew)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.actOpen)
        self.fileMenu.addAction(self.actOpenComfyWorkflow)
        self.fileMenu.addAction(self.actSave)
        self.fileMenu.addAction(self.actSaveAs)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.actExit)

    def createEditMenu(self):
        menubar = self.menuBar()
        self.editMenu = menubar.addMenu('&Edit')
        self.editMenu.addAction(self.actUndo)
        self.editMenu.addAction(self.actRedo)
        self.editMenu.addSeparator()
        self.editMenu.addAction(self.actCut)
        self.editMenu.addAction(self.actCopy)
        self.editMenu.addAction(self.actPaste)
        self.editMenu.addSeparator()
        self.editMenu.addAction(self.actDelete)

    def setTitle(self):
        """Function responsible for setting window title"""
        title = "Node Editor - "
        title += self.getCurrentNodeEditorWidget().getUserFriendlyFilename()

        self.setWindowTitle(title)


    def closeEvent(self, event):
        """Handle close event. Ask before we loose work"""
        if self.maybeSave():
            event.accept()
        else:
            event.ignore()

    def isModified(self) -> bool:
        """Has current :class:`~nodeeditor.node_scene.Scene` been modified?

        :return: ``True`` if current :class:`~nodeeditor.node_scene.Scene` has been modified
        :rtype: ``bool``
        """
        nodeeditor = self.getCurrentNodeEditorWidget()
        return nodeeditor.scene.isModified() if nodeeditor else False

    def getCurrentNodeEditorWidget(self) -> NodeEditorWidget:
        """get current :class:`~nodeeditor.node_editor_widget`

        :return: get current :class:`~nodeeditor.node_editor_widget`
        :rtype: :class:`~nodeeditor.node_editor_widget`
        """
        return self.centralWidget()

    def maybeSave(self) -> bool:
        """If current `Scene` is modified, ask a dialog to save the changes. Used before
        closing window / mdi child document

        :return: ``True`` if we can continue in the `Close Event` and shutdown. ``False`` if we should cancel
        :rtype: ``bool``
        """
        if not self.isModified():
            return True

        res = QMessageBox.warning(self, "About to loose your work?",
                "The document has been modified.\n Do you want to save your changes?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
              )

        if res == QMessageBox.Save:
            return self.onFileSave()
        elif res == QMessageBox.Cancel:
            return False

        return True


    def onScenePosChanged(self, x:int, y:int):
        """Handle event when cursor position changed on the `Scene`

        :param x: new cursor x position
        :type x:
        :param y: new cursor y position
        :type y:
        """
        self.status_mouse_pos.setText("Scene Pos: [%d, %d]" % (x, y))

    def getFileDialogDirectory(self):
        """Returns starting directory for ``QFileDialog`` file open/save"""
        return ''

    def getFileDialogFilter(self):
        """Returns ``str`` standard file open/save filter for ``QFileDialog``"""
        return 'Graph (*.json);;All files (*)'

    def onFileNew(self):
        """Hande File New operation"""
        if self.maybeSave():
            self.getCurrentNodeEditorWidget().fileNew()
            self.setTitle()


    def onFileOpen(self):
        """Handle File Open operation"""
        if self.maybeSave():
            fname, filter = QFileDialog.getOpenFileName(self, 'Open graph from file', self.getFileDialogDirectory(), self.getFileDialogFilter())
            if fname != '' and os.path.isfile(fname):
                self.getCurrentNodeEditorWidget().fileLoad(fname)
                self.setTitle()

    def onFileSave(self):
        """Handle File Save operation"""
        current_nodeeditor = self.getCurrentNodeEditorWidget()
        if current_nodeeditor is not None:
            if not current_nodeeditor.isFilenameSet(): return self.onFileSaveAs()

            current_nodeeditor.fileSave()
            self.statusBar().showMessage("Successfully saved %s" % current_nodeeditor.filename, 5000)

            # support for MDI app
            if hasattr(current_nodeeditor, "setTitle"): current_nodeeditor.setTitle()
            else: self.setTitle()
            return True

    def onFileSaveAs(self):
        """Handle File Save As operation"""
        current_nodeeditor = self.getCurrentNodeEditorWidget()
        if current_nodeeditor is not None:
            fname, filter = QFileDialog.getSaveFileName(self, 'Save graph to file', self.getFileDialogDirectory(), self.getFileDialogFilter())
            if fname == '': return False

            self.onBeforeSaveAs(current_nodeeditor, fname)
            current_nodeeditor.fileSave(fname)
            self.statusBar().showMessage("Successfully saved as %s" % current_nodeeditor.filename, 5000)

            # support for MDI app
            if hasattr(current_nodeeditor, "setTitle"): current_nodeeditor.setTitle()
            else: self.setTitle()
            return True

    def onBeforeSaveAs(self, current_nodeeditor: 'NodeEditorWidget', filename: str):
        """
        Event triggered after choosing filename and before actual fileSave(). We are passing current_nodeeditor because
        we will loose focus after asking with QFileDialog and therefore getCurrentNodeEditorWidget will return None
        """
        pass

    def onEditUndo(self):
        """Handle Edit Undo operation"""
        if self.getCurrentNodeEditorWidget():
            self.getCurrentNodeEditorWidget().scene.history.undo()

    def onEditRedo(self):
        """Handle Edit Redo operation"""
        if self.getCurrentNodeEditorWidget():
            self.getCurrentNodeEditorWidget().scene.history.redo()

    def onEditDelete(self):
        """Handle Delete Selected operation"""
        if self.getCurrentNodeEditorWidget():
            self.getCurrentNodeEditorWidget().scene.getView().deleteSelected()

    def onEditCut(self):
        """Handle Edit Cut to clipboard operation"""
        if self.getCurrentNodeEditorWidget():
            data = self.getCurrentNodeEditorWidget().scene.clipboard.serializeSelected(delete=True)
            str_data = json.dumps(data, indent=4)
            QApplication.instance().clipboard().setText(str_data)

    def onEditCopy(self):
        """Handle Edit Copy to clipboard operation"""
        if self.getCurrentNodeEditorWidget():
            data = self.getCurrentNodeEditorWidget().scene.clipboard.serializeSelected(delete=False)
            str_data = json.dumps(data, indent=4)
            QApplication.instance().clipboard().setText(str_data)

    def onEditPaste(self):
        """Handle Edit Paste from clipboard operation"""
        if self.getCurrentNodeEditorWidget():
            raw_data = QApplication.instance().clipboard().text()

            try:
                data = json.loads(raw_data)
            except ValueError as e:
                print("Pasting of not valid json data!", e)
                return

            # check if the json data are correct
            if 'nodes' not in data:
                print("JSON does not contain any nodes!")
                return

            return self.getCurrentNodeEditorWidget().scene.clipboard.deserializeFromClipboard(data)

    def readSettings(self):
        """Read the permanent profile settings for this app"""
        settings = QSettings(self.name_company, self.name_product)
        pos = settings.value('pos', QPoint(200, 200))
        size = settings.value('size', QSize(400, 400))
        self.move(pos)
        self.resize(size)

    def writeSettings(self):
        """Write the permanent profile settings for this app"""
        settings = QSettings(self.name_company, self.name_product)
        settings.setValue('pos', self.pos())
        settings.setValue('size', self.size())
    # New method to handle opening a Comfy workflow
    def onOpenComfyWorkflow(self):
        """Handle the action of opening a Comfy workflow, converting it to Ainodes format, and loading it."""
        if self.maybeSave():
            fname, _ = QFileDialog.getOpenFileName(
                self,
                'Open Comfy Workflow',
                self.getFileDialogDirectory(),
                'Comfy Workflow (*.json);;All files (*)'
            )
            if fname != '' and os.path.isfile(fname):
                # Read the Comfy JSON file
                with open(fname, 'r') as f:
                    comfy_data = json.load(f)
                # Convert Comfy JSON to Ainodes JSON
                ainodes_data = self.convertComfyToAinodes(comfy_data)
                # Load Ainodes JSON into the node editor
                with open('temp.json', "w") as file:
                    file.write(json.dumps(ainodes_data, indent=4))
                self.getCurrentNodeEditorWidget().fileLoad('temp.json')
                self.setTitle()

    def convertComfyToAinodes(self, comfy_data):
        """Convert Comfy JSON data to Ainodes JSON format.

        :param comfy_data: Dictionary loaded from Comfy JSON
        :return: Ainodes data dictionary
        """
        # Initialize the Ainodes data dictionary
        ainodes_data = {
            "id": 1,
            "scene_width": 64000,
            "scene_height": 64000,
            "nodes": [],
            "edges": []
        }

        node_id_map = {}
        socket_id_map = {}
        next_node_id = 1
        next_socket_id = 1000

        # Mapping from Comfy types to Ainodes socket_type
        type_to_socket_type = {
            "EXEC": 1,
            "LATENT": 2,
            "CONDITIONING": 3,
            "PIPE": 4,
            "MODEL": 4,
            "VAE": 4,
            "CLIP": 4,
            "COGVIDEOPIPE": 4,
            "IMAGE": 5,
            "DATA": 6,
            "STRING": 7,
            "INT": 8,
            "FLOAT": 9,
            # Any other types default to 7 (STRING)
        }

        # Process nodes
        for comfy_node in comfy_data['nodes']:
            comfy_node_id = comfy_node['id']
            ainodes_node_id = next_node_id
            node_id_map[comfy_node_id] = ainodes_node_id
            next_node_id += 1

            # Get position
            pos_dict = comfy_node['pos']
            pos_x = pos_dict.get("0", 0)
            pos_y = pos_dict.get("1", 0)

            ainodes_node = {
                "id": ainodes_node_id,
                "title": comfy_node['type'],
                "pos_x": pos_x,
                "pos_y": pos_y,
                "inputs": [],
                "outputs": [],
                "content": {},
                "op_code": 0,
                "content_label_objname": comfy_node['type'].lower()
            }

            custom_input_socket_names = []
            custom_output_socket_names = []
            input_socket_types = []
            output_socket_types = []

            # Process inputs
            for input_socket in comfy_node.get('inputs', []):
                socket_id = next_socket_id
                next_socket_id += 1
                comfy_socket_type = input_socket['type']
                socket_type = type_to_socket_type.get(comfy_socket_type, 7)  # default to 7 (STRING)
                input_dict = {
                    "id": socket_id,
                    "index": input_socket.get('slot_index', 0),
                    "multi_edges": False,
                    "position": 3,
                    "socket_type": socket_type,
                    "name": input_socket['name']
                }
                socket_id_map[(comfy_node_id, 'input', input_socket.get('slot_index', 0))] = socket_id
                ainodes_node['inputs'].append(input_dict)

                custom_input_socket_names.append(input_socket['name'])
                input_socket_types.append(socket_type)

            # Process outputs
            for output_socket in comfy_node.get('outputs', []):
                socket_id = next_socket_id
                next_socket_id += 1
                comfy_socket_type = output_socket['type']
                socket_type = type_to_socket_type.get(comfy_socket_type, 7)  # default to 7 (STRING)
                output_dict = {
                    "id": socket_id,
                    "index": output_socket.get('slot_index', 0),
                    "multi_edges": True,
                    "position": 6,
                    "socket_type": socket_type,
                    "name": output_socket['name']
                }
                socket_id_map[(comfy_node_id, 'output', output_socket.get('slot_index', 0))] = socket_id
                ainodes_node['outputs'].append(output_dict)

                custom_output_socket_names.append(output_socket['name'])
                output_socket_types.append(socket_type)

            # Map content (properties and widgets_values)
            content = {}
            content['properties'] = comfy_node.get('properties', {})
            content['widgets_values'] = comfy_node.get('widgets_values', [])
            content['custom_input_socket_names'] = custom_input_socket_names
            content['custom_output_socket_names'] = custom_output_socket_names
            content['_inputs'] = input_socket_types
            content['_outputs'] = output_socket_types
            # Include other node parameters as needed

            ainodes_node['content'] = content

            ainodes_data['nodes'].append(ainodes_node)

        # Process links
        for link in comfy_data.get('links', []):
            # Link format: [link_id, from_node_id, from_socket_index, to_node_id, to_socket_index, "TYPE"]
            if len(link) < 6:
                continue  # Skip invalid links
            link_id, from_node_id, from_socket_index, to_node_id, to_socket_index, _ = link

            from_socket_id = socket_id_map.get((from_node_id, 'output', from_socket_index))
            to_socket_id = socket_id_map.get((to_node_id, 'input', to_socket_index))

            if from_socket_id is None or to_socket_id is None:
                continue  # Skip if sockets are not found

            ainodes_edge = {
                "id": link_id,
                "edge_type": 5,
                "start": from_socket_id,
                "end": to_socket_id
            }

            ainodes_data['edges'].append(ainodes_edge)

        return ainodes_data