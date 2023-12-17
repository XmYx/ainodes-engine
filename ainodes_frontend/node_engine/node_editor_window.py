# -*- coding: utf-8 -*-
"""
A module containing the Main Window class
"""
import json
import os
from functools import partial

from qtpy import QtWidgets
from qtpy.QtCore import QSize, QSettings, QPoint
from qtpy.QtWidgets import QMainWindow, QLabel, QAction, QMessageBox, QFileDialog, QApplication

from ainodes_frontend import singleton as gs
from ainodes_frontend.base.open_os_browser import open_folder_in_file_browser
from ainodes_frontend.node_engine.node_editor_widget import NodeEditorWidget


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
        """Set up this ``QMainWindow``. Create :class:`~node_engine.node_editor_widget.NodeEditorWidget`, Actions and Menus"""
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
        self.statusBar().showMessage("aiNodes Ready")
        self.status_mouse_pos = QLabel("")

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


    def createMenus(self):
        """Create Menus for `File` and `Edit`"""
        self.createFileMenu()
        self.createGraphsMenu()
        self.createEditMenu()

    def add_files_to_menu(self, dir_path, menu):
        for item in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item)
            if os.path.isdir(item_path):
                # If it's a directory, create a submenu and recurse
                sub_menu = menu.addMenu(item)
                self.add_files_to_menu(item_path, sub_menu)
            elif item.endswith('.json'):
                # If it's a JSON file, create an action and add to the current menu
                file_action = QAction(item, self)
                file_action.triggered.connect(partial(self.onFileOpenAction, item_path))
                menu.addAction(file_action)

    def createGraphsMenu(self):
        menubar = self.menuBar()
        self.graphsMenu = menubar.addMenu('&Graphs')
        self.graphsSubMenu = QtWidgets.QMenu('Graphs', self)


        if os.path.isdir('graphs'):

            self.add_files_to_menu('graphs', self.graphsSubMenu)

            # # List all JSON files in the graphs and example_graphs folders
            # graphs_files = [f for f in os.listdir('graphs') if f.endswith('.json')]
            # #example_graphs_files = [f for f in os.listdir('example_graphs') if f.endswith('.json')]
            #
            # # Add JSON files to the submenus and connect actions
            # for file in graphs_files:
            #     file_action = QAction(file, self)
            #     file_action.triggered.connect(partial(self.onFileOpenAction, os.path.join('graphs', file)))
            #     self.graphsSubMenu.addAction(file_action)

        directory_path = 'ai_nodes'
        if os.path.isdir(directory_path):
            custom_nodes = get_dir_content(directory_path)
        else:
            custom_nodes = None
        self.exampleGraphsSubMenu = QtWidgets.QMenu('Example Graphs', self)
        if custom_nodes is not None:
            for folder in custom_nodes:
                folder_path = os.path.join('ai_nodes', folder, 'resources', 'examples')

                if "__pycache__" not in folder_path:
                    if os.path.isdir(folder_path):
                        folder_submenu = self.exampleGraphsSubMenu.addMenu(folder)
                        example_files = os.listdir(folder_path)

                        for file in example_files:
                            file_path = os.path.join(folder_path, file)
                            if os.path.isfile(file_path):
                                file_action = QAction(file, self)
                                file_action.triggered.connect(partial(self.onFileOpenAction, file_path))
                                folder_submenu.addAction(file_action)
                            elif os.path.isdir(file_path):
                                folder_submenu_ = folder_submenu.addMenu(file)
                                example_files_ = os.listdir(file_path)

                                for file_ in example_files_:
                                    file_path_ = os.path.join(file_path, file_)
                                    if os.path.isfile(file_path_):
                                        file_action_ = QAction(file_, self)
                                        file_action_.triggered.connect(partial(self.onFileOpenAction, file_path_))
                                        folder_submenu_.addAction(file_action_)

        # Add submenus to the File menu
        self.graphsMenu.addMenu(self.graphsSubMenu)
        self.graphsMenu.addMenu(self.exampleGraphsSubMenu)


    def createFileMenu(self):
        menubar = self.menuBar()
        self.fileMenu = menubar.addMenu('&File')

        # Create submenus for graphs and example_graphs

        # Add other actions to the File menu
        self.fileMenu.addAction(self.actNew)
        self.fileMenu.addSeparator()
        if hasattr(gs, 'prefs'):
            self.defaultDirsSubMenu = QtWidgets.QMenu('User Directories', self)
            default_dirs = {"Stills":"output/stills",
                            "MP4s":"output/mp4s",
                            "SD Models":gs.prefs.checkpoints,
                            "VAE":gs.prefs.vae,
                            "ControlNet":gs.prefs.controlnet,
                            "Embeddings":gs.prefs.embeddings,
                            "Upscalers":gs.prefs.upscalers,
                            "LORAs":gs.prefs.loras,
                            "T2I Adapters":gs.prefs.t2i_adapter}

            for dir_name, dir in default_dirs.items():
                dir_action = QAction(dir_name, self)
                dir_action.triggered.connect(partial(open_folder_in_file_browser, os.path.join(os.getcwd(), dir)))
                self.defaultDirsSubMenu.addAction(dir_action)

            self.fileMenu.addMenu(self.defaultDirsSubMenu)
        else:
            print("Warning, PATHs were not set up")
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.actOpen)
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
        """Has current :class:`~node_engine.node_scene.Scene` been modified?

        :return: ``True`` if current :class:`~node_engine.node_scene.Scene` has been modified
        :rtype: ``bool``
        """
        nodeeditor = self.getCurrentNodeEditorWidget()
        return nodeeditor.scene.isModified() if nodeeditor else False

    def getCurrentNodeEditorWidget(self) -> NodeEditorWidget:
        """get current :class:`~node_engine.node_editor_widget`

        :return: get current :class:`~node_engine.node_editor_widget`
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
        return
        #self.status_mouse_pos.setText("Scene Pos: [%d, %d]" % (x, y))

    def getFileDialogDirectory(self):
        directory = os.getcwd()
        #directory = f"{directory}/example_graphs"
        """Returns starting directory for ``QFileDialog`` file open/save"""
        return directory

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
            fname, filter = QFileDialog.getOpenFileName(None, 'Open graph from file', f"{self.getFileDialogDirectory()}/graphs", self.getFileDialogFilter())
            if fname != '' and os.path.isfile(fname):
                self.getCurrentNodeEditorWidget().fileLoad(fname)
                self.setTitle()
    def onFileOpenAction(self, fname):
        self.onFileNew()
        self.getCurrentNodeEditorWidget().fileLoad(fname)
        self.getCurrentNodeEditorWidget().setWindowTitle(fname)
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

        print()

        """Handle File Save As operation"""
        current_nodeeditor = self.getCurrentNodeEditorWidget()
        if current_nodeeditor is not None:
            if current_nodeeditor.json_name:
                if "Subgraph" in current_nodeeditor.json_name:
                    dest = "subgraphs"
            else:
                dest = "graphs"
            fname, filter = QFileDialog.getSaveFileName(self, 'Save graph to file', f"{self.getFileDialogDirectory()}/{dest}", self.getFileDialogFilter())
            if fname == '': return False
            self.onBeforeSaveAs(current_nodeeditor, fname)
            current_nodeeditor.fileSave(fname)
            self.window().nodesListWidget.addMyItems()
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

def get_dir_content(dir):
    content = []

    # Iterate over each item in the directory
    for item in os.listdir(dir):
        item_path = os.path.join(dir, item)

        # Check if the item is a directory
        if os.path.isdir(item_path):
            content.append(item)
    return content