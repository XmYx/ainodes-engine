import datetime
import re
import sys
sys.path.extend(['src/ainodes'])
sys.path.extend(['src/flux-fp8-api'])

from PyQt5.QtGui import QSurfaceFormat, QOpenGLContext
# Enable OpenGL globally
def enable_opengl():
    fmt = QSurfaceFormat()
    fmt.setRenderableType(QSurfaceFormat.OpenGL)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    fmt.setVersion(3, 3)  # Requesting OpenGL 3.3
    QSurfaceFormat.setDefaultFormat(fmt)

# Call the function to enable OpenGL
enable_opengl()

from PyQt5.QtCore import pyqtSignal, QSettings

from node_core.console import NodesConsole, StreamRedirect

start_time = datetime.datetime.now()
print(f"Start aiNodes, please wait. {start_time}")
from PyQt5.QtWidgets import QApplication, QLabel, QHBoxLayout, QSlider

# sys.path.extend(['src/pyqt-node-editor'])

#
from PyQt5.QtCore import Qt, QSignalBlocker
from PyQt5.QtGui import QCloseEvent
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QAction, QPlainTextEdit, QInputDialog, QSizePolicy, \
    QComboBox, QWidgetAction, QToolBar, QTableWidget
import PyQtAds as QtAds

import os
from PyQt5.QtGui import QIcon, QKeySequence
from PyQt5.QtWidgets import QMdiArea, QWidget, QDockWidget, QAction, QMessageBox, QFileDialog
from PyQt5.QtCore import Qt, QSignalMapper

from nodeeditor.utils import loadStylesheets
from nodeeditor.node_editor_window import NodeEditorWindow
from node_core.calc_sub_window import CalculatorSubWindow
from node_core.calc_drag_listbox import QDMDragListbox
from nodeeditor.utils import dumpException, pp
from node_core.node_register import NODE_CLASSES
from ainodes_core.settings_dialog import SettingsDialog
# Enabling edge validators
from nodeeditor.node_edge import Edge
from nodeeditor.node_edge_validators import (
    edge_validator_debug,
    edge_cannot_connect_two_outputs_or_two_inputs,
    edge_cannot_connect_input_and_output_of_same_node
)
Edge.registerEdgeValidator(edge_validator_debug)
Edge.registerEdgeValidator(edge_cannot_connect_two_outputs_or_two_inputs)
Edge.registerEdgeValidator(edge_cannot_connect_input_and_output_of_same_node)

# images for the dark skin
# import examples.example_calculator.qss.nodeeditor_dark_resources

DEBUG = False


class CalculatorWindow(NodeEditorWindow):

    refresh_nodes_signal = pyqtSignal()

    def initUI(self):

        self.settings = QSettings("aiNodes", "Engine")  # Adjust these values
        # Load ComfyUI nodes if path is set
        comfyui_path = self.settings.value("comfyui_path", "")
        if comfyui_path:
            try:
                from base_nodes.readapt_nodes.readapt import register_comfy_nodes
                register_comfy_nodes(comfyui_path)
            except:
                pass
        self.default_font_size = self.font().pointSizeF()
        # Load and store the original stylesheet
        with open('qss/nodeeditor-dark.qss', 'r') as file:
            self.original_qss = file.read()
            self.setStyleSheet(self.original_qss)

        self.init_scale_slider()

        self.name_company = 'aiNodes'
        self.name_product = 'Engine'

        self.stylesheet_filename = os.path.join(os.path.dirname(__file__), "qss/nodeeditor-dark.qss")
        loadStylesheets(
            os.path.join(os.path.dirname(__file__), "qss/nodeeditor-dark.qss"),
            self.stylesheet_filename
        )
        self.empty_icon = QIcon(".")

        if DEBUG:
            print("Registered nodes:")
            pp(NODE_CLASSES)
        QtAds.CDockManager.setConfigFlag(QtAds.CDockManager.OpaqueSplitterResize, True)
        QtAds.CDockManager.setConfigFlag(QtAds.CDockManager.XmlCompressionEnabled, False)
        QtAds.CDockManager.setConfigFlag(QtAds.CDockManager.FocusHighlighting, True)
        self.dock_manager = QtAds.CDockManager(self)

        self.mdiArea = QMdiArea()
        self.mdiArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.mdiArea.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.mdiArea.setViewMode(QMdiArea.TabbedView)
        self.mdiArea.setDocumentMode(True)
        self.mdiArea.setTabsClosable(True)
        self.mdiArea.setTabsMovable(True)
        #self.setCentralWidget(self.mdiArea)
        central_dock_widget = QtAds.CDockWidget("CentralWidget")
        central_dock_widget.setWidget(self.mdiArea)
        central_dock_area = self.dock_manager.setCentralWidget(central_dock_widget)
        central_dock_area.setAllowedAreas(QtAds.DockWidgetArea.OuterDockAreas)
        self.mdiArea.subWindowActivated.connect(self.updateMenus)
        self.windowMapper = QSignalMapper(self)
        self.windowMapper.mapped[QWidget].connect(self.setActiveSubWindow)
        self.setup_model_manager()

        self.createNodesDock()

        self.createActions()
        self.createMenus()
        self.createToolBars()
        self.createStatusBar()
        self.updateMenus()

        self.readSettings()
        self.setWindowTitle("aiNodes 2.0")
        self.create_console_widget()
        self.create_perspective_ui()
        # self.save_last_perspective()
        self.load_last_perspective()
    def scale_stylesheet(self, qss, scale_factor):
        # Regular expression to find numeric values in the QSS
        pattern = re.compile(r'(\d+)(px|pt)?')

        def replace_func(match):
            value = int(match.group(1))
            unit = match.group(2) or ''
            scaled_value = int(value * scale_factor)
            return f'{scaled_value}{unit}'

        scaled_qss = pattern.sub(replace_func, qss)
        return scaled_qss

    def update_scaling(self, value):
        scale_factor = value / 100.0

        # Update the application font
        font = self.font()
        font.setPointSizeF(self.default_font_size * scale_factor)
        self.setFont(font)

        # Optionally, update the stylesheet with new scaling
        scaled_qss = self.scale_stylesheet(self.original_qss, scale_factor)
        self.setStyleSheet(scaled_qss)

    def init_scale_slider(self):
        # Create a widget to hold the slider and label
        scale_widget = QWidget()
        scale_layout = QHBoxLayout()
        scale_widget.setLayout(scale_layout)

        # Create a label for the slider
        self.scale_label = QLabel('Scale:')
        scale_layout.addWidget(self.scale_label)

        # Create the slider
        self.scale_slider = QSlider(Qt.Horizontal)
        self.scale_slider.setMinimum(50)   # 50%
        self.scale_slider.setMaximum(200)  # 200%
        self.scale_slider.setValue(100)    # Default 100%
        self.scale_slider.setTickInterval(10)
        self.scale_slider.setTickPosition(QSlider.TicksBelow)
        scale_layout.addWidget(self.scale_slider)

        # Add the scale widget to the status bar or main layout
        self.statusBar().addPermanentWidget(scale_widget)

        # Connect the slider to the scaling function
        self.scale_slider.valueChanged.connect(self.update_scaling)

    def setup_model_manager(self):

        # model_manager.load_model(
        #     (
        #         '/home/mix/Playground/ComfyUI/models/clip/clip_l.safetensors',
        #         '/home/mix/Playground/ComfyUI/models/clip/t5xxl_fp8_e4m3fn.safetensors'
        #     ),
        #     '/home/mix/Playground/ComfyUI/models/vae/flux-ae.safetensors',
        #     '/home/mix/Playground/ComfyUI/models/unet/flux1-dev.safetensors'
        # )
        # model_manager_list = ModelManagerListWidget(model_manager)
        # model_manager.refresh_list = model_manager_list.refresh_list_signal
        # model_manager.refresh_nodes = self.refresh_nodes_signal
        # self.model_manager_dock = QtAds.CDockWidget("Console")
        # self.model_manager_dock.setWidget(model_manager_list)
        # self.model_manager_dock.setMinimumSizeHintMode(QtAds.CDockWidget.MinimumSizeHintFromDockWidget)
        # self.model_manager_dock.setWindowTitle("Console")
        # self.dock_manager.addDockWidget(QtAds.DockWidgetArea.LeftDockWidgetArea, self.model_manager_dock)

        pass

        # model_manager_list.show()
    def create_perspective_ui(self):
        save_perspective_action = QAction("Create Perspective", self)
        save_perspective_action.triggered.connect(self.save_perspective)
        perspective_list_action = QWidgetAction(self)
        self.perspective_combobox = QComboBox(self)
        self.perspective_combobox.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.perspective_combobox.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.perspective_combobox.textActivated.connect(self.dock_manager.openPerspective)
        perspective_list_action.setDefaultWidget(self.perspective_combobox)
        self.toolBar = QToolBar("toolBar", self)
        self.toolBar.addSeparator()
        self.toolBar.addAction(perspective_list_action)
        self.toolBar.addAction(save_perspective_action)

        self.addToolBar(Qt.TopToolBarArea, self.toolBar)
        self.toolBar.setMovable(True)

    def create_console_widget(self):
        # Create a text widget for stdout and stderr
        self.text_widget = NodesConsole()
        # Set up the StreamRedirect objects
        self.stdout_redirect = StreamRedirect()
        self.stderr_redirect = StreamRedirect()
        self.stdout_redirect.text_written.connect(self.text_widget.write)
        self.stderr_redirect.text_written.connect(self.text_widget.write)
        sys.stdout = self.stdout_redirect
        sys.stderr = self.stderr_redirect

        self.console = QtAds.CDockWidget("Console")
        self.console.setWidget(self.text_widget)
        self.console.setMinimumSizeHintMode(QtAds.CDockWidget.MinimumSizeHintFromDockWidget)
        self.console.setWindowTitle("Console")
        self.dock_manager.addDockWidget(QtAds.DockWidgetArea.LeftDockWidgetArea, self.console)

        # widget = QtWidgets.QWidget()
        # layout = QtWidgets.QHBoxLayout(widget)
        # layout.setContentsMargins(5,5,5,5)
        # layout.addWidget(self.text_widget)
        #self.addDockWidget(Qt.LeftDockWidgetArea, self.console)
    def save_perspective(self):
        perspective_name, ok = QInputDialog.getText(self, "Save Perspective", "Enter Unique name:")
        if not ok or not perspective_name:
            return

        self.dock_manager.addPerspective(perspective_name)
        blocker = QSignalBlocker(self.perspective_combobox)
        self.perspective_combobox.clear()
        self.perspective_combobox.addItems(self.dock_manager.perspectiveNames())
        self.perspective_combobox.setCurrentText(perspective_name)
    def closeEvent(self, event):
        self.save_last_perspective()

        self.mdiArea.closeAllSubWindows()
        if self.mdiArea.currentSubWindow():
            event.ignore()
        else:
            self.writeSettings()
            event.accept()
            # hacky fix for PyQt 5.14.x
            import sys
            sys.exit(0)

    def load_last_perspective(self):
        settings = QSettings("Settings.ini", QSettings.IniFormat)
        self.dock_manager.loadPerspectives(settings)
        self.perspective_combobox.addItems(self.dock_manager.perspectiveNames())

        # self.perspective_combo_box.clear()
        # self.perspective_combo_box.addItems(self.dock_manager.perspectiveNames())
        # Automatically select the "last_perspective" if it exists
        if "last_perspective" in self.dock_manager.perspectiveNames():
        #     self.perspective_combobox.setCurrentText("last_perspective")
            self.dock_manager.openPerspective("last_perspective")

    def save_last_perspective(self):
        perspective_name = "last_perspective"
        self.dock_manager.addPerspective(perspective_name)
        settings = QSettings("Settings.ini", QSettings.IniFormat)
        self.dock_manager.savePerspectives(settings)

        # perspective_data = self.dock_manager.saveState().toJson().data().decode()
        # workspace_dir = os.path.join(os.path.dirname(__file__), "workspaces")
        # os.makedirs(workspace_dir, exist_ok=True)
        # perspective_file = os.path.join(workspace_dir, "last_perspective.json")
        # with open(perspective_file, "w") as f:
        #     f.write(perspective_data)

    def createActions(self):
        super().createActions()

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

    def onFileNew(self):
        try:
            subwnd = self.createMdiChild()
            subwnd.widget().fileNew()
            subwnd.show()
        except Exception as e: dumpException(e)


    def onFileOpen(self):
        fnames, filter = QFileDialog.getOpenFileNames(self, 'Open graph from file', self.getFileDialogDirectory(), self.getFileDialogFilter())

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
        QMessageBox.about(self, "About Calculator NodeEditor Example",
                "The <b>Calculator NodeEditor</b> example demonstrates how to write multiple "
                "document interface applications using qtpy and NodeEditor. For more information visit: "
                "<a href='https://www.blenderfreak.com/'>www.BlenderFreak.com</a>")

    def open_settings_dialog(self):
        settings_dialog = SettingsDialog(self)
        if settings_dialog.exec_():
            # Settings were saved; now attempt to register ComfyUI nodes
            comfyui_path = settings_dialog.settings.value("comfyui_path", "")
            if comfyui_path:
                try:
                    from base_nodes.readapt_nodes.readapt import register_comfy_nodes
                    register_comfy_nodes(comfyui_path)
                except:
                    # Import failed, try to install requirements and retry
                    import subprocess
                    import sys
                    import os

                    requirements_file = os.path.join(comfyui_path, 'requirements.txt')
                    if os.path.exists(requirements_file):
                        try:
                            # Install requirements
                            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_file])
                            # Retry the import after installing requirements
                            from base_nodes.readapt_nodes.readapt import register_comfy_nodes
                            register_comfy_nodes(comfyui_path)
                        except:
                            pass
    def createMenus(self):
        super().createMenus()

        self.windowMenu = self.menuBar().addMenu("&Window")
        self.updateWindowMenu()
        self.windowMenu.aboutToShow.connect(self.updateWindowMenu)

        self.menuBar().addSeparator()

        self.helpMenu = self.menuBar().addMenu("&Help")
        self.helpMenu.addAction(self.actAbout)

        self.editMenu.aboutToShow.connect(self.updateEditMenu)
        self.actSettings = QAction("&Settings", self, statusTip="Open Settings", triggered=self.open_settings_dialog)
        self.settingsMenu = self.menuBar().addMenu("&Settings")
        self.settingsMenu.addAction(self.actSettings)
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
        #toolbar_nodes.setChecked(self.nodesDock.isVisible())

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
        self.nodesListWidget = QDMDragListbox(self)

        self.nodesDock = QtAds.CDockWidget("Table 1")
        self.nodesDock.setWidget(self.nodesListWidget)
        self.nodesDock.setMinimumSizeHintMode(QtAds.CDockWidget.MinimumSizeHintFromDockWidget)
        self.nodesDock.resize(250, 150)
        self.nodesDock.setMinimumSize(200, 150)

        self.dock_manager.addDockWidget(QtAds.DockWidgetArea.LeftDockWidgetArea, self.nodesDock)
        
        self.refresh_nodes_signal.connect(self.nodesListWidget.addMyItems)

    def createStatusBar(self):
        self.statusBar().showMessage("Ready")

    def createMdiChild(self, child_widget=None):
        nodeeditor = child_widget if child_widget is not None else CalculatorSubWindow()
        subwnd = self.mdiArea.addSubWindow(nodeeditor)
        subwnd.setWindowIcon(self.empty_icon)
        # nodeeditor.scene.addItemSelectedListener(self.updateEditMenu)
        # nodeeditor.scene.addItemsDeselectedListener(self.updateEditMenu)
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

if __name__ == '__main__':

    # from tests.qss_editor_window import MyApplication, QSSLiveEditor

    app = QApplication(sys.argv)
    #
    # #Add QSS Debugger
    # qss_debugger = QSSLiveEditor(app, 'qss/nodeeditor-dark.qss')

    # print(QStyleFactory.keys())
    #
    # app.setStyle('Fusion')
    # qdarktheme.setup_theme()


    wnd = CalculatorWindow()

    wnd.stylesheet_filename = os.path.join(os.path.dirname(__file__), 'qss/nodeeditor-dark.qss')

    loadStylesheets(
        wnd.stylesheet_filename,
        wnd.stylesheet_filename
    )

    wnd.show()

    #Show QSS Debugger
    # qss_debugger.show()

    end_time = datetime.datetime.now()
    print(f"Initialization took: {end_time - start_time}")
    sys.exit(app.exec_())
