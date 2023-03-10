import os
from subprocess import run, PIPE

import requests
from qtpy import QtWidgets, QtCore, QtGui
from qtpy.QtGui import QIcon, QKeySequence
from qtpy.QtWidgets import QMdiArea, QDockWidget, QAction, QMessageBox, QFileDialog
from qtpy.QtCore import Qt, QSignalMapper

from ainodes_frontend.base.worker import Worker
from ainodes_frontend.node_engine.utils import loadStylesheets
from ainodes_frontend.node_engine.node_editor_window import NodeEditorWindow
from ainodes_frontend.base.node_sub_window import CalculatorSubWindow
from ainodes_frontend.base.ai_nodes_listbox import QDMDragListbox
from ainodes_frontend.node_engine.utils_no_qt import dumpException, pp
from ainodes_frontend.base.node_config import CALC_NODES, import_nodes_from_file, import_nodes_from_subdirectories

# Enabling edge validators
from ainodes_frontend.node_engine.node_edge import Edge
from ainodes_frontend.node_engine.node_edge_validators import (
    edge_validator_debug,
    edge_cannot_connect_two_outputs_or_two_inputs,
    edge_cannot_connect_input_and_output_of_same_node,
    edge_cannot_connect_input_and_output_of_different_type

)
Edge.registerEdgeValidator(edge_validator_debug)
Edge.registerEdgeValidator(edge_cannot_connect_two_outputs_or_two_inputs)
Edge.registerEdgeValidator(edge_cannot_connect_input_and_output_of_same_node)
Edge.registerEdgeValidator(edge_cannot_connect_input_and_output_of_different_type)


from pyqtgraph.console import ConsoleWidget

# images for the dark skin
DEBUG = False
from ainodes_frontend import singleton as gs

gs.loaded_models = {}
gs.models = {}

class StdoutTextEdit(QtWidgets.QPlainTextEdit):
    signal = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Set the maximum block count to 1000
        self.setMaximumBlockCount(1000)
        self.signal.connect(self.write_function)

    def write(self, text):
        self.signal.emit(text)

    def flush(self):
        pass  # no-op, since we're not buffering

    @QtCore.Slot(str)
    def write_function(self, text):
        # Split the text into lines
        lines = text.splitlines()

        # Append the lines without adding extra line breaks
        cursor = self.textCursor()
        for line in lines:
            cursor.movePosition(QtGui.QTextCursor.End)
            cursor.insertText(line)
            cursor.insertBlock()

        # Scroll to the bottom
        self.ensureCursorVisible()

def remove_empty_lines(file_path):
    with open(file_path, "r+") as f:
        content = f.readlines()
        if not content:
            # if the file is empty, add the base node as the first line
            f.write("XmYx/ainodes_engine_base_nodes\n")
        else:
            if not content[0].strip().startswith("XmYx/ainodes_engine_base_nodes"):
                # if the first line doesn't start with the base node, add it as the first line
                f.seek(0)
                f.write("XmYx/ainodes_engine_base_nodes\n")
                f.writelines(line for line in content if line.strip())
                f.truncate()
            else:
                # if the first line is already the base node, remove empty lines
                f.seek(0)
                f.writelines(line for line in content if line.strip())
                f.truncate()
class GitHubRepositoriesDialog(QtWidgets.QDockWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.parent = parent
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Node Packages")

        main_widget = QtWidgets.QWidget(self)

        layout = QtWidgets.QHBoxLayout(main_widget)
        self.setWidget(main_widget)
        self.list_widget = QtWidgets.QListWidget()

        layout.addWidget(self.list_widget)

        repository_widget = QtWidgets.QWidget()
        repository_layout = QtWidgets.QVBoxLayout(repository_widget)
        repository_label = QtWidgets.QLabel("Node Packages")
        self.repository_name_label = QtWidgets.QLabel()
        self.repository_url_label = QtWidgets.QLabel()
        self.repository_icon_label = QtWidgets.QLabel()
        self.download_button = QtWidgets.QPushButton("Download")
        self.download_button.clicked.connect(self.download_repository)
        self.update_button = QtWidgets.QPushButton("Update / Import")
        self.update_button.clicked.connect(self.update_repository)
        self.update_button.hide()

        repository_layout.addWidget(repository_label)
        repository_layout.addWidget(self.repository_icon_label)
        repository_layout.addWidget(self.repository_name_label)
        repository_layout.addWidget(self.repository_url_label)
        repository_layout.addWidget(self.download_button)
        repository_layout.addWidget(self.update_button)
        layout.addWidget(repository_widget)

        self.list_widget.currentRowChanged.connect(self.show_repository_details)
        self.list_widget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self.right_click_menu)

        self.add_repository_action = QtWidgets.QAction("Add Repository", self)
        self.add_repository_action.triggered.connect(self.add_repository)
        self.delete_repository_action = QtWidgets.QAction("Delete Repository", self)
        self.delete_repository_action.triggered.connect(self.delete_repository)

        self.context_menu = QtWidgets.QMenu(self)
        self.context_menu.addAction(self.add_repository_action)
        self.context_menu.addAction(self.delete_repository_action)

        self.repository_icon_label.setFixedSize(128, 128)
        self.repolist = "repositories.txt"


    def load_repositories(self):
        remove_empty_lines(self.repolist)
        with open("repositories.txt") as f:
            repositories = f.read().splitlines()
        for repository in repositories:
            item = QtWidgets.QListWidgetItem(repository)
            folder = repository.split("/")[1]
            if os.path.isdir(f"./custom_nodes/{folder}"):
                item.setBackground(Qt.darkGreen)
                item.setForeground(Qt.white)
                self.update_button.setVisible(True)
                self.download_button.setVisible(False)
            else:
                item.setForeground(Qt.black)
                item.setBackground(Qt.darkYellow)
            self.list_widget.addItem(item)

    def show_repository_details(self, row):
        repository = self.list_widget.item(row).text()
        folder = repository.split("/")[1]
        self.repository_name_label.setText(repository)
        url = f"https://github.com/{repository}"
        self.repository_url_label.setText(url)
        icon_url = f"https://raw.githubusercontent.com/{repository}/main/icon.png"
        icon_pixmap = QtGui.QPixmap()
        icon_pixmap.loadFromData(requests.get(icon_url).content)
        self.repository_icon_label.setPixmap(icon_pixmap)
        self.repository_icon_label.setScaledContents(True)
        if os.path.isdir(f"./custom_nodes/{folder}"):
            self.list_widget.currentItem().setBackground(Qt.darkGreen)
            self.update_button.show()
            self.download_button.hide()
        else:
            self.list_widget.currentItem().setBackground(Qt.darkYellow)
            self.update_button.hide()
            self.download_button.show()

    def download_repository(self):
        worker = Worker(self.download_repository_thread)
        worker.signals.result.connect(self.download_repository_finished)
        self.parent.threadpool.start(worker)
    def download_repository_thread(self, progress_callback=None):
        repository = self.repository_name_label.text()
        folder = repository.split("/")[1]
        command = f"git clone https://github.com/{repository} ./custom_nodes/{folder} && pip install -r ./custom_nodes/{folder}/requirements.txt"
        result = run(command, shell=True, stdout=self.parent.text_widget, stderr=self.parent.text_widget)
        return result
    @QtCore.Slot(object)
    def download_repository_finished(self, result):
        repository = self.repository_name_label.text()
        folder = repository.split("/")[1]
        if result.returncode == 0:
            import_nodes_from_subdirectories(f"./custom_nodes/{folder}")
            self.parent.nodesListWidget.addMyItems()
            QtWidgets.QMessageBox.information(self, "Download Complete", f"{repository} was downloaded successfully.")
            self.list_widget.currentItem().setBackground(Qt.darkGreen)
            self.update_button.hide()
        else:
            QtWidgets.QMessageBox.critical(self, "Download Failed",
            f"An error occurred while downloading {repository}:\n{result.stderr.decode()}")

    def update_repository(self):
        worker = Worker(self.update_repository_thread)
        worker.signals.result.connect(self.update_repository_finished)
        self.parent.threadpool.start(worker)

    def update_repository_thread(self, progress_callback=None):
        repository = self.repository_name_label.text()
        folder = repository.split("/")[1]
        #command = f"git -C ./custom_nodes/{folder} stash && git -C ./custom_nodes/{folder} pull && pip install -r ./custom_nodes/{folder}/requirements.txt"
        command = f"git -C ./custom_nodes/{folder} pull && pip install -r ./custom_nodes/{folder}/requirements.txt"
        result = run(command, shell=True, stdout=self.parent.text_widget, stderr=self.parent.text_widget)
        return result
    @QtCore.Slot(object)
    def update_repository_finished(self, result):
        repository = self.repository_name_label.text()
        folder = repository.split("/")[1]
        if result.returncode == 0:
            import_nodes_from_subdirectories(f"./custom_nodes/{folder}")
            self.parent.nodesListWidget.addMyItems()
            QtWidgets.QMessageBox.information(self, "Update Complete", f"{repository} was updated successfully.")
            self.list_widget.currentItem().setBackground(Qt.darkGreen)
            self.update_button.hide()
        else:
            QtWidgets.QMessageBox.critical(self, "Update Failed",
                                 f"An error occurred while updating {repository}:\n{result.stderr.decode()}")

    def add_repository(self):
        remove_empty_lines(self.repolist)
        text, ok = QtWidgets.QInputDialog.getText(self, "Add Repository",
                                                  "Enter the repository name (e.g. owner/repo):")
        if ok and text:
            with open("repositories.txt", "r") as f:
                lines = f.read().splitlines()
                if text in lines:
                    QtWidgets.QMessageBox.warning(self, "Duplicate Repository", f"{text} already exists in the list.")
                    return
            with open("repositories.txt", "a") as f:
                if lines:
                    f.write("\n")
                f.write(text)
            item = QtWidgets.QListWidgetItem(text)
            item.setForeground(Qt.black)
            item.setBackground(Qt.darkYellow)
            self.list_widget.addItem(item)

    def delete_repository(self):
        remove_empty_lines(self.repolist)
        row = self.list_widget.currentRow()
        if row != -1:
            repository = self.list_widget.item(row).text()
            if repository == "XmYx/ainodes_engine_base_nodes":
                QtWidgets.QMessageBox.warning(self, "Default Repository", "The default repository cannot be deleted.")
                return
            with open("repositories.txt", "r") as f:
                lines = f.readlines()
            with open("repositories.txt", "w") as f:
                for line in lines:
                    if line.strip() != repository:
                        f.write(line)
            self.list_widget.takeItem(row)
    def right_click_menu(self, position):
        self.context_menu.exec_(self.list_widget.mapToGlobal(position))
class CalculatorWindow(NodeEditorWindow):

    def __init__(self):
        super().__init__()

        # Create a text widget for stdout and stderr
        self.text_widget = ConsoleWidget()
        #self.text_widget2 = ConsoleWidget()

        self.threadpool = QtCore.QThreadPool()
        # Create a dock widget for the text widget and add it to the main window
        self.dock_widget = QDockWidget('Output', self)
        self.dock_widget.setAllowedAreas(Qt.BottomDockWidgetArea)
        self.layout = QtWidgets.QHBoxLayout()
        self.layout.setContentsMargins(5,5,5,5)

        #layout.addWidget(self.text_widget2)
        self.dock_widget.setWidget(self.text_widget)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.dock_widget)
        # Redirect stdout and stderr to the text widget

    def initUI(self):

        self.name_company = 'aiNodes'
        self.name_product = 'AI Node Editor'
        gs.loaded_models["loaded"] = []

        """self.stylesheet_filename = os.path.join(os.path.dirname(__file__), "ainodes_frontend/qss/nodeeditor-dark.qss")
        loadStylesheets(
            os.path.join(os.path.dirname(__file__), "ainodes_frontend/qss/nodeeditor-dark.qss"),
            self.stylesheet_filename
        )"""

        self.empty_icon = QIcon("")

        if DEBUG:
            print("Registered nodes:")
            pp(CALC_NODES)


        self.mdiArea = QMdiArea()
        self.mdiArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.mdiArea.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.mdiArea.setViewMode(QMdiArea.TabbedView)
        self.mdiArea.setDocumentMode(True)
        self.mdiArea.setTabsClosable(True)
        self.mdiArea.setTabsMovable(True)
        self.setCentralWidget(self.mdiArea)

        self.mdiArea.subWindowActivated.connect(self.updateMenus)
        self.windowMapper = QSignalMapper(self)
        self.windowMapper.mappedInt[int].connect(self.setActiveSubWindow)

        # Connect the signal mapper to the map slot using a lambda function
        self.windowMapper.mappedInt[int].connect(lambda id: self.setActiveSubWindow(self.subWindowList()[id]))

        self.createNodesDock()

        self.createActions()
        self.createMenus()
        self.createToolBars()
        self.createStatusBar()
        self.updateMenus()

        self.readSettings()

        self.setWindowTitle("aiNodes - Engine")

        self.show_github_repositories()

    def show_github_repositories(self):
        self.node_packages = GitHubRepositoriesDialog(self)
        self.node_packages.load_repositories()
        self.addDockWidget(Qt.BottomDockWidgetArea, self.node_packages)
        #dialog.show()

    def closeEvent(self, event):
        self.mdiArea.closeAllSubWindows()
        if self.mdiArea.currentSubWindow():
            event.ignore()
        else:
            self.writeSettings()
            event.accept()
            # hacky fix for PyQt 5.14.x
            import sys
            sys.exit(0)


    def createActions(self):
        super().createActions()
        self.actNode = QAction('&Add Node', self, shortcut='Ctrl+L', statusTip="Open new node", triggered=self.onNodeOpen)
        self.actNodePacks = QAction('&Node Packages', self, shortcut='Ctrl+K', statusTip="Download Nodes", triggered=self.show_github_repositories)


        self.actClose = QAction("Cl&ose", self, statusTip="Close the active window", triggered=self.mdiArea.closeActiveSubWindow)
        self.actCloseAll = QAction("Close &All", self, statusTip="Close all the windows", triggered=self.mdiArea.closeAllSubWindows)
        self.actTile = QAction("&Tile", self, statusTip="Tile the windows", triggered=self.mdiArea.tileSubWindows)
        self.actCascade = QAction("&Cascade", self, statusTip="Cascade the windows", triggered=self.mdiArea.cascadeSubWindows)
        self.actNext = QAction("Ne&xt", self, shortcut=QKeySequence.NextChild, statusTip="Move the focus to the next window", triggered=self.mdiArea.activateNextSubWindow)
        self.actPrevious = QAction("Pre&vious", self, shortcut=QKeySequence.PreviousChild, statusTip="Move the focus to the previous window", triggered=self.mdiArea.activatePreviousSubWindow)

        self.actSeparator = QAction(self)
        self.actSeparator.setSeparator(True)

        self.actAbout = QAction("&About", self, statusTip="Show the application's About box", triggered=self.about)


    def getCurrentNodeEditorWidget(self):
        """ we're returning NodeEditorWidget here... """
        activeSubWindow = self.mdiArea.activeSubWindow()
        if activeSubWindow:
            return activeSubWindow.widget()
        return None
    def onNodeOpen(self):
        """Handle File Open operation"""
        fname, filter = QFileDialog.getOpenFileName(None, 'Open graph from file', f"{self.getFileDialogDirectory()}/ainodes_frontend/custom_nodes", 'Python Files (*.py)')
        if fname != '' and os.path.isfile(fname):
            import_nodes_from_file(fname)
            self.nodesListWidget.addMyItems()

    def onFileNew(self):
        try:
            subwnd = self.createMdiChild()
            subwnd.widget().fileNew()
            subwnd.show()
        except Exception as e: dumpException(e)


    def onFileOpen(self):
        fnames, filter = QFileDialog.getOpenFileNames(self, 'Open graph from file', f"{self.getFileDialogDirectory()}/graphs", self.getFileDialogFilter())

        try:
            for fname in fnames:
                if fname:
                    existing = self.findMdiChild(fname)
                    if existing:
                        self.mdiArea.setActiveSubWindow(existing)
                    else:
                        # we need to create new subWindow and open the file
                        nodeeditor = CalculatorSubWindow()
                        if nodeeditor.fileLoad(fname):
                            self.statusBar().showMessage("File %s loaded" % fname, 5000)
                            nodeeditor.setTitle()
                            subwnd = self.createMdiChild(nodeeditor)
                            subwnd.show()
                        else:
                            nodeeditor.close()
        except Exception as e: dumpException(e)


    def about(self):
        QMessageBox.about(self, "About aiNodes engine",
                "<b>aiNodes engine</b> v0.1 "
                "for more information, please visit the GitHub Page"
                "<a href='https://www.github.com/XmYx/ainodes-engine'>aiNodes Engine GitHub</a>")

    def createMenus(self):
        super().createMenus()

        self.windowMenu = self.menuBar().addMenu("&Window")
        self.updateWindowMenu()
        self.windowMenu.aboutToShow.connect(self.updateWindowMenu)

        self.menuBar().addSeparator()

        self.helpMenu = self.menuBar().addMenu("&Help")
        self.helpMenu.addAction(self.actAbout)

        self.editMenu.aboutToShow.connect(self.updateEditMenu)

        self.fileMenu.addAction(self.actNode)
        self.fileMenu.addAction(self.actNodePacks)
        # Get the index of the action in the fileMenu
        action_index = self.fileMenu.actions().index(self.actNode)

        # Insert the action at the new index
        new_index = max(0, action_index - 4)
        self.fileMenu.insertAction(self.fileMenu.actions()[new_index], self.actNode)
        # Get the index of the action in the fileMenu
        action_index = self.fileMenu.actions().index(self.actNodePacks)

        # Insert the action at the new index
        new_index = max(0, action_index - 4)
        self.fileMenu.insertAction(self.fileMenu.actions()[new_index], self.actNodePacks)
    def updateMenus(self):
        # print("update Menus")
        active = self.getCurrentNodeEditorWidget()
        hasMdiChild = (active is not None)

        self.actSave.setEnabled(hasMdiChild)
        self.actSaveAs.setEnabled(hasMdiChild)
        self.actClose.setEnabled(hasMdiChild)
        self.actCloseAll.setEnabled(hasMdiChild)
        self.actTile.setEnabled(hasMdiChild)
        self.actCascade.setEnabled(hasMdiChild)
        self.actNext.setEnabled(hasMdiChild)
        self.actPrevious.setEnabled(hasMdiChild)
        self.actSeparator.setVisible(hasMdiChild)

        self.updateEditMenu()

    def updateEditMenu(self):
        try:
            # print("update Edit Menu")
            active = self.getCurrentNodeEditorWidget()
            hasMdiChild = (active is not None)

            self.actPaste.setEnabled(hasMdiChild)

            self.actCut.setEnabled(hasMdiChild and active.hasSelectedItems())
            self.actCopy.setEnabled(hasMdiChild and active.hasSelectedItems())
            self.actDelete.setEnabled(hasMdiChild and active.hasSelectedItems())

            self.actUndo.setEnabled(hasMdiChild and active.canUndo())
            self.actRedo.setEnabled(hasMdiChild and active.canRedo())
        except Exception as e: dumpException(e)



    def updateWindowMenu(self):
        self.windowMenu.clear()

        toolbar_nodes = self.windowMenu.addAction("Nodes Toolbar")
        toolbar_nodes.setCheckable(True)
        toolbar_nodes.triggered.connect(self.onWindowNodesToolbar)
        toolbar_nodes.setChecked(self.nodesDock.isVisible())

        self.windowMenu.addSeparator()

        self.windowMenu.addAction(self.actClose)
        self.windowMenu.addAction(self.actCloseAll)
        self.windowMenu.addSeparator()
        self.windowMenu.addAction(self.actTile)
        self.windowMenu.addAction(self.actCascade)
        self.windowMenu.addSeparator()
        self.windowMenu.addAction(self.actNext)
        self.windowMenu.addAction(self.actPrevious)
        self.windowMenu.addAction(self.actSeparator)

        windows = self.mdiArea.subWindowList()
        self.actSeparator.setVisible(len(windows) != 0)

        for i, window in enumerate(windows):
            child = window.widget()

            text = "%d %s" % (i + 1, child.getUserFriendlyFilename())
            if i < 9:
                text = '&' + text

            action = self.windowMenu.addAction(text)
            action.setCheckable(True)
            action.setChecked(child is self.getCurrentNodeEditorWidget())
            action.triggered.connect(self.windowMapper.map)
            self.windowMapper.setMapping(action, window)

    def onWindowNodesToolbar(self):
        if self.nodesDock.isVisible():
            self.nodesDock.hide()
        else:
            self.nodesDock.show()

    def createToolBars(self):
        pass

    def createNodesDock(self):
        self.nodesListWidget = QDMDragListbox()

        self.nodesDock = QDockWidget("Nodes")
        self.nodesDock.setWidget(self.nodesListWidget)
        self.nodesDock.setFloating(False)

        self.addDockWidget(Qt.LeftDockWidgetArea, self.nodesDock)

    def createStatusBar(self):
        self.statusBar().showMessage("Ready")

    def createMdiChild(self, child_widget=None):
        nodeeditor = child_widget if child_widget is not None else CalculatorSubWindow()
        subwnd = self.mdiArea.addSubWindow(nodeeditor)
        subwnd.setWindowIcon(self.empty_icon)



        # node_engine.scene.addItemSelectedListener(self.updateEditMenu)
        # node_engine.scene.addItemsDeselectedListener(self.updateEditMenu)
        nodeeditor.scene.history.addHistoryModifiedListener(self.updateEditMenu)
        nodeeditor.addCloseEventListener(self.onSubWndClose)
        return subwnd

    def onSubWndClose(self, widget, event):
        existing = self.findMdiChild(widget.filename)
        self.mdiArea.setActiveSubWindow(existing)

        if self.maybeSave():
            event.accept()
        else:
            event.ignore()


    def findMdiChild(self, filename):
        for window in self.mdiArea.subWindowList():
            if window.widget().filename == filename:
                return window
        return None


    def setActiveSubWindow(self, window):
        if window:
            self.mdiArea.setActiveSubWindow(window)