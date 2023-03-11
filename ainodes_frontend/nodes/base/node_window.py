import os
import sys
import threading

from qtpy import QtWidgets, QtCore, QtGui
from qtpy.QtGui import QIcon, QKeySequence
from qtpy.QtWidgets import QMdiArea, QDockWidget, QAction, QMessageBox, QFileDialog
from qtpy.QtCore import Qt, QSignalMapper

from ainodes_backend.node_engine.utils import loadStylesheets
from ainodes_backend.node_engine.node_editor_window import NodeEditorWindow
from ainodes_frontend.nodes.base.node_sub_window import CalculatorSubWindow
from ainodes_frontend.nodes.base.ai_nodes_listbox import QDMDragListbox
from ainodes_backend.node_engine.utils_no_qt import dumpException, pp
from ainodes_frontend.nodes.base.node_config import CALC_NODES, import_nodes_from_file

# Enabling edge validators
from ainodes_backend.node_engine.node_edge import Edge
from ainodes_backend.node_engine.node_edge_validators import (
    edge_validator_debug,
    edge_cannot_connect_two_outputs_or_two_inputs,
    edge_cannot_connect_input_and_output_of_same_node,
    edge_cannot_connect_input_and_output_of_different_type

)
Edge.registerEdgeValidator(edge_validator_debug)
Edge.registerEdgeValidator(edge_cannot_connect_two_outputs_or_two_inputs)
Edge.registerEdgeValidator(edge_cannot_connect_input_and_output_of_same_node)
Edge.registerEdgeValidator(edge_cannot_connect_input_and_output_of_different_type)


# images for the dark skin
DEBUG = False
from ainodes_backend import singleton as gs

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
class CalculatorWindow(NodeEditorWindow):

    def __init__(self):
        super().__init__()

        # Create a text widget for stdout and stderr
        self.text_widget = StdoutTextEdit()

        # Create a dock widget for the text widget and add it to the main window
        self.dock_widget = QDockWidget('Output', self)
        self.dock_widget.setAllowedAreas(Qt.BottomDockWidgetArea)
        self.dock_widget.setWidget(self.text_widget)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.dock_widget)

        # Redirect stdout and stderr to the text widget
        sys.stdout = self.text_widget
        sys.stderr = self.text_widget

    def initUI(self):

        self.name_company = 'aiNodes'
        self.name_product = 'AI Node Editor'
        gs.loaded_models["loaded"] = []
        #print(gs.loaded_models)
        #print(gs.loaded_models["loaded"])

        self.stylesheet_filename = os.path.join(os.path.dirname(__file__), "qss/node_engine-dark.qss")
        loadStylesheets(
            os.path.join(os.path.dirname(__file__), "qss/node_engine-dark.qss"),
            self.stylesheet_filename
        )

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
        # Get the index of the action in the fileMenu
        action_index = self.fileMenu.actions().index(self.actNode)

        # Insert the action at the new index
        new_index = max(0, action_index - 4)
        self.fileMenu.insertAction(self.fileMenu.actions()[new_index], self.actNode)
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

        self.addDockWidget(Qt.RightDockWidgetArea, self.nodesDock)

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